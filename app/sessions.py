"""Durable per-session storage for batched operation submission.

A session pins the shared initial document and initial version vector and
accumulates every accepted batch into one cumulative operation log.  Both
reads and batch commits re-run the ordinary merge rules *from the initial
document* over the whole log -- the previous result is never used as a new
baseline -- so late-arriving concurrent writes still take part in the
original arbitration and conflict accounting, and a batch may causally
reference operations stored by earlier batches.

Batch commits are atomic: validation, recomputation and the revision bump
happen under a single lock, and the updated session is written to disk
(temp file + ``os.replace``) before the lock is released.  Two concurrent
submissions carrying the same expected revision therefore cannot both
succeed -- the loser observes the bumped revision and gets a conflict.

Sessions are persisted as one JSON file per session inside a directory
(``SESSION_STORE_DIR`` env var, default ``.sessions``), so they remain
readable after the service (or ``docker compose``) restarts.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import merge_logs, validate_and_dedupe
from .models import Operation

_SESSION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")


class SessionNotFoundError(KeyError):
    """Raised when a session id does not exist."""


class RevisionConflictError(RuntimeError):
    """Raised when the expected revision does not match the current one."""


@dataclass
class SessionRecord:
    """The durable state of one session.

    ``document``/``version`` are the pinned initial state; ``log`` is the
    cumulative, de-duplicated operation log accepted so far; ``revision``
    counts the batches that contributed at least one new operation.
    """

    session_id: str
    revision: int
    document: Any
    version: dict[str, int]
    log: list[Operation] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "document": self.document,
            "version": self.version,
            "log": [op.model_dump(mode="json") for op in self.log],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=data["session_id"],
            revision=int(data["revision"]),
            document=data["document"],
            version={k: int(v) for k, v in data["version"].items()},
            log=[Operation.model_validate(o) for o in data["log"]],
        )


@dataclass
class BatchOutcome:
    """Result of an accepted batch submission."""

    session_id: str
    revision: int
    accepted: int
    duplicates: int


class SessionStore:
    """Directory-backed session store with atomic batch commits."""

    def __init__(self, directory: str | os.PathLike[str]):
        self._dir = Path(directory)
        self._lock = threading.RLock()
        self._cache: dict[str, SessionRecord] = {}

    @classmethod
    def from_env(cls) -> "SessionStore":
        return cls(os.environ.get("SESSION_STORE_DIR", ".sessions"))

    # -- persistence ---------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _persist(self, rec: SessionRecord) -> None:
        """Write ``rec`` atomically (temp file + rename within the directory)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / f".{rec.session_id}.{uuid.uuid4().hex}.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(rec.to_json(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._path(rec.session_id))

    def _load(self, session_id: str) -> SessionRecord | None:
        if not _SESSION_ID_RE.fullmatch(session_id):
            return None
        rec = self._cache.get(session_id)
        if rec is not None:
            return rec
        try:
            raw = self._path(session_id).read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            rec = SessionRecord.from_json(json.loads(raw))
        except (KeyError, TypeError, ValueError):
            return None
        self._cache[session_id] = rec
        return rec

    # -- session lifecycle ----------------------------------------------

    def create(self, document: Any, version: dict[str, int]) -> SessionRecord:
        """Create a session pinned to ``document``/``version`` at revision 0."""
        initial_version = dict(version or {})
        validate_and_dedupe([], initial_version)  # rejects negative entries
        with self._lock:
            rec = SessionRecord(
                session_id=uuid.uuid4().hex,
                revision=0,
                document=document,
                version=initial_version,
            )
            self._persist(rec)
            self._cache[rec.session_id] = rec
            return rec

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            rec = self._load(session_id)
        if rec is None:
            raise SessionNotFoundError(session_id)
        return rec

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Recompute the merged state from the initial document and full log."""
        with self._lock:
            rec = self._load(session_id)
            if rec is None:
                raise SessionNotFoundError(session_id)
            result = merge_logs(rec.document, rec.version, [rec.log])
            return {
                "session_id": rec.session_id,
                "revision": rec.revision,
                "document": result["document"],
                "version": result["version"],
                "conflicts": result["conflicts"],
                "tombstones": result["tombstones"],
            }

    def submit_batch(
        self, session_id: str, expected_revision: int, operations: list[Operation]
    ) -> BatchOutcome:
        """Validate, recompute and commit one batch atomically.

        The batch is validated together with the already-stored log under
        the ordinary rules: retransmitted operations (same id, same
        content) are not stored twice, the same id with different content
        is rejected, and causal references may point at stored operations.
        A batch that contributes no new operation leaves the revision
        untouched; any validation or execution failure leaves the session
        untouched as well (no partial commits).
        """
        with self._lock:
            rec = self._load(session_id)
            if rec is None:
                raise SessionNotFoundError(session_id)
            if expected_revision != rec.revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision} but session "
                    f"{session_id!r} is at revision {rec.revision}"
                )
            combined = list(rec.log) + list(operations)
            # Full validation of the cumulative log: identical
            # retransmissions are de-duplicated, conflicting ids rejected.
            unique_ops, duplicates, _ = validate_and_dedupe(combined, rec.version)
            stored = {(op.actor, op.seq) for op in rec.log}
            new_ops = [op for op in unique_ops if (op.actor, op.seq) not in stored]
            if not new_ops:
                return BatchOutcome(session_id, rec.revision, 0, duplicates)
            # Recompute from the initial document over the cumulative log;
            # a validation or execution failure raises before anything is
            # written, so the session keeps its previous revision and log.
            merge_logs(rec.document, rec.version, [unique_ops])
            rec.log = unique_ops
            rec.revision += 1
            self._persist(rec)
            return BatchOutcome(session_id, rec.revision, len(new_ops), duplicates)
