"""Durable storage for offline replica sessions with batched log delivery.

A session holds the common initial document and initial version vector plus
every operation log received so far.  Each accepted non-empty batch raises
the session's ``revision`` by exactly one (a batch whose operations are all
exact retransmissions is a no-op that keeps the revision).

Read state is always recomputed from scratch from the initial document and
the *whole* accumulated log with the engine's ordinary merge rules -- the
previous result is never used as a new baseline, so a late concurrent write
arriving in a later batch still takes part in the original arbitration and
conflict accounting, and a cross-batch causal reference may point at an
operation stored by an earlier batch.

Persistence is a single SQLite database (default ``./data/sessions.db``,
overridable through ``SESSION_DB_PATH``).  Batch submission runs inside one
``BEGIN IMMEDIATE`` transaction: revision check, duplicate detection, full
log validation, recomputation and the revision insert commit atomically --
two requests submitting against the same revision serialize on the write
lock, the first commits and the second fails with :class:`StaleRevision`,
so exactly one of them can succeed.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from typing import Any

from .engine import LogValidationError, canonical_operation, merge_logs
from .models import Operation


class SessionNotFound(LookupError):
    """Raised when no session exists for the given id (HTTP 404)."""


class StaleRevision(Exception):
    """Raised when a batch assumes a revision older than the current one."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected revision {expected} but the session is at revision {current}"
        )
        self.expected = expected
        self.current = current


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    document    TEXT NOT NULL,
    version     TEXT NOT NULL,
    revision    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS operations (
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    ord         INTEGER NOT NULL,
    actor       TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (session_id, actor, seq)
);
CREATE INDEX IF NOT EXISTS operations_order
    ON operations (session_id, ord);
"""


class SessionStore:
    """SQLite-backed durable session store."""

    def __init__(self, path: str | None = None):
        path = path or os.environ.get("SESSION_DB_PATH", "data/sessions.db")
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    # -- connection management -------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # ``check_same_thread=False`` is safe because each call opens its own
        # short-lived connection and SQLite serializes writers on the file.
        conn = sqlite3.connect(
            self.path, timeout=15, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            # executescript commits any pending transaction first, so run
            # the idempotent DDL without an explicit surrounding transaction
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _load_ops(conn: sqlite3.Connection, session_id: str) -> list[Operation]:
        rows = conn.execute(
            "SELECT payload FROM operations WHERE session_id = ? ORDER BY ord",
            (session_id,),
        ).fetchall()
        return [Operation.model_validate_json(row["payload"]) for row in rows]

    @staticmethod
    def _recompute(
        document: Any, version: dict[str, int], ops: list[Operation]
    ) -> dict[str, Any]:
        """Re-run the full merge from the initial document (never a baseline)."""
        return merge_logs(document, version, [ops])

    # -- public API --------------------------------------------------------

    def create_session(
        self, document: Any, version: dict[str, int] | None
    ) -> tuple[str, int]:
        """Create a session at revision 0 holding the common initial state."""
        initial_version = dict(version or {})
        for actor, v in initial_version.items():
            if v < 0:
                raise LogValidationError(
                    f"initial version for {actor!r} is negative"
                )
        session_id = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO sessions (id, document, version, revision) "
                "VALUES (?, ?, ?, 0)",
                (
                    session_id,
                    json.dumps(document, default=str),
                    json.dumps(initial_version),
                ),
            )
        finally:
            conn.close()
        return session_id, 0

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Return recomputed state plus the current revision for a session."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT document, version, revision FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise SessionNotFound(session_id)
            ops = self._load_ops(conn, session_id)
        finally:
            conn.close()
        document = json.loads(row["document"])
        version = json.loads(row["version"])
        result = self._recompute(document, version, ops)
        return {
            "session_id": session_id,
            "revision": row["revision"],
            "document": result["document"],
            "version": result["version"],
            "conflicts": result["conflicts"],
            "tombstones": result["tombstones"],
        }

    def append_batch(
        self,
        session_id: str,
        expected_revision: int,
        incoming: list[Operation],
    ) -> dict[str, int]:
        """Validate and append one non-empty batch atomically.

        Same-id same-content operations (also within the batch) are skipped;
        same-id different-content, an invalid accumulated log or an execution
        failure raise :class:`LogValidationError` and nothing is written.
        An outdated ``expected_revision`` raises :class:`StaleRevision`.
        """
        conn = self._connect()
        done = False
        try:
            # one atomic unit: take the write lock, check the revision,
            # validate against the accumulated log, recompute, then commit
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT document, version, revision FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                done = True
                raise SessionNotFound(session_id)
            revision = row["revision"]
            if expected_revision != revision:
                conn.execute("ROLLBACK")
                done = True
                raise StaleRevision(expected_revision, revision)

            stored = self._load_ops(conn, session_id)
            stored_keys = {(o.actor, o.seq): canonical_operation(o) for o in stored}

            new_ops: list[Operation] = []
            seen_in_batch: dict[tuple[str, int], str] = {}
            duplicates = 0
            for op in incoming:
                key = (op.actor, op.seq)
                canonical = canonical_operation(op)
                if key in stored_keys:
                    if stored_keys[key] != canonical:
                        conn.execute("ROLLBACK")
                        done = True
                        raise LogValidationError(
                            f"operation {op.actor}:{op.seq} appears twice with "
                            f"different payloads"
                        )
                    duplicates += 1
                    continue
                if key in seen_in_batch:
                    if seen_in_batch[key] != canonical:
                        conn.execute("ROLLBACK")
                        done = True
                        raise LogValidationError(
                            f"operation {op.actor}:{op.seq} appears twice with "
                            f"different payloads"
                        )
                    duplicates += 1
                    continue
                seen_in_batch[key] = canonical
                new_ops.append(op)

            # full validation + recomputation from the initial document; late
            # concurrent writes and cross-batch causal refs are all re-adjudicated
            document = json.loads(row["document"])
            version = json.loads(row["version"])
            try:
                self._recompute(document, version, stored + new_ops)
            except LogValidationError:
                conn.execute("ROLLBACK")
                done = True
                raise

            if new_ops:
                max_ord = conn.execute(
                    "SELECT COALESCE(MAX(ord), 0) AS m FROM operations "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["m"]
                rows = [
                    (
                        session_id,
                        max_ord + i + 1,
                        op.actor,
                        op.seq,
                        canonical_operation(op),
                    )
                    for i, op in enumerate(new_ops)
                ]
                conn.executemany(
                    "INSERT INTO operations "
                    "(session_id, ord, actor, seq, payload) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.execute(
                    "UPDATE sessions SET revision = ? WHERE id = ?",
                    (revision + 1, session_id),
                )
            conn.execute("COMMIT")
            done = True
        except Exception:
            if not done:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        finally:
            conn.close()
        return {
            "revision": revision + 1 if new_ops else revision,
            "accepted": len(new_ops),
            "duplicates": duplicates,
        }


# ---------------------------------------------------------------------------
# process-wide registry (keyed by database path)
# ---------------------------------------------------------------------------

_stores: dict[str, SessionStore] = {}
_stores_lock = threading.Lock()


def get_store(path: str | None = None) -> SessionStore:
    """Return the shared store for ``path`` (default: SESSION_DB_PATH env)."""
    path = path or os.environ.get("SESSION_DB_PATH", "data/sessions.db")
    with _stores_lock:
        store = _stores.get(path)
        if store is None:
            store = SessionStore(path)
            _stores[path] = store
        return store
