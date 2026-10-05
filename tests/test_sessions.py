"""End-to-end tests for durable offline sessions with batched log delivery."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app import store as store_mod
from app.main import app
from app.models import Operation

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_session_db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite file for every test."""
    db_path = str(tmp_path / "sessions.db")
    monkeypatch.setenv("SESSION_DB_PATH", db_path)
    store_mod._stores.clear()
    yield
    store_mod._stores.clear()


def op(actor, seq, version, path, action, value=None, index=None):
    body = {
        "actor": actor,
        "seq": seq,
        "version": version,
        "path": path,
        "action": action,
        "value": value,
    }
    if index is not None:
        body["index"] = index
    return body


def opm(*args, **kwargs):
    """Same as op() but as an Operation instance (for direct store calls)."""
    return Operation.model_validate(op(*args, **kwargs))


def create(document=None, version=None):
    return client.post(
        "/sessions", json={"document": document, "version": version or {}}
    )


def submit(session_id, expected_revision, operations):
    return client.post(
        f"/sessions/{session_id}/batches",
        json={
            "expected_revision": expected_revision,
            "operations": operations,
        },
    )


def view(session_id):
    return client.get(f"/sessions/{session_id}")


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_create_returns_revision_zero():
    r = create({"title": "draft"})
    assert r.status_code == 201
    body = r.json()
    assert body["revision"] == 0
    assert body["session_id"]
    got = view(body["session_id"])
    assert got.status_code == 200
    assert got.json() == {
        "session_id": body["session_id"],
        "revision": 0,
        "document": {"title": "draft"},
        "version": {},
        "conflicts": [],
        "tombstones": [],
    }


def test_create_preserves_initial_version_vector():
    r = create({"k": 1}, {"alice": 3, "bob": 1})
    assert r.status_code == 201
    body = view(r.json()["session_id"]).json()
    assert body["version"] == {"alice": 3, "bob": 1}


def test_create_with_negative_initial_version_rejected():
    r = create({}, {"alice": -1})
    assert r.status_code == 422


def test_get_unknown_session_404():
    assert view("deadbeef").status_code == 404


def test_batch_unknown_session_404():
    r = submit(
        "deadbeef", 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)]
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# batched delivery & revision accounting
# ---------------------------------------------------------------------------


def test_batches_accumulate_and_revision_increments():
    sid = create({"count": 0}).json()["session_id"]
    r1 = submit(sid, 0, [op("alice", 1, {"alice": 1}, "/count", "set", 1)])
    assert r1.status_code == 200
    assert r1.json() == {
        "session_id": sid,
        "revision": 1,
        "accepted": 1,
        "duplicates": 0,
    }
    r2 = submit(
        sid,
        1,
        [op("alice", 2, {"alice": 2}, "/count", "set", 2)],
    )
    assert r2.status_code == 200
    assert r2.json()["revision"] == 2
    body = view(sid).json()
    assert body["revision"] == 2
    assert body["document"] == {"count": 2}
    assert body["version"] == {"alice": 2}


def test_retransmission_is_idempotent_and_does_not_increment_revision():
    sid = create({}).json()["session_id"]
    o1 = op("alice", 1, {"alice": 1}, "/x", "set", 1)
    assert submit(sid, 0, [o1]).json()["revision"] == 1
    # exact same op again -> duplicate, revision stays 1
    r = submit(sid, 1, [o1])
    assert r.status_code == 200
    assert r.json() == {
        "session_id": sid,
        "revision": 1,
        "accepted": 0,
        "duplicates": 1,
    }
    # a mixed batch: one duplicate + one new op still advances the revision
    o2 = op("alice", 2, {"alice": 2}, "/y", "set", 2)
    r = submit(sid, 1, [o1, o2])
    assert r.status_code == 200
    assert r.json()["accepted"] == 1
    assert r.json()["duplicates"] == 1
    assert r.json()["revision"] == 2
    body = view(sid).json()
    assert body["document"] == {"x": 1, "y": 2}
    assert body["version"] == {"alice": 2}


def test_intra_batch_duplicate_is_deduped():
    sid = create({}).json()["session_id"]
    o1 = op("alice", 1, {"alice": 1}, "/x", "set", 1)
    r = submit(sid, 0, [o1, dict(o1)])
    assert r.status_code == 200
    assert r.json()["accepted"] == 1
    assert r.json()["duplicates"] == 1
    assert r.json()["revision"] == 1


def test_empty_batch_is_rejected():
    sid = create({}).json()["session_id"]
    r = submit(sid, 0, [])
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# recomputation from the initial document, never from the previous result
# ---------------------------------------------------------------------------


def test_late_concurrent_write_participates_in_arbitration():
    sid = create({"title": "draft"}).json()["session_id"]
    # batch 1: alice writes /title
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "alice-v")])
    # batch 2: bob's *concurrent* write arrives late; ("bob",1) must win the
    # original arbitration even though it lands after alice's batch
    r = submit(sid, 1, [op("bob", 1, {"bob": 1}, "/title", "set", "bob-v")])
    assert r.status_code == 200
    body = view(sid).json()
    assert body["document"]["title"] == "bob-v"
    assert len(body["conflicts"]) == 1
    assert body["conflicts"][0]["path"] == "/title"
    assert body["conflicts"][0]["winner"]["actor"] == "bob"


def test_late_concurrent_write_equal_effects_no_conflict():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    submit(sid, 1, [op("bob", 1, {"bob": 1}, "/x", "set", 1)])
    body = view(sid).json()
    assert body["document"] == {"x": 1}
    assert body["conflicts"] == []


def test_cross_batch_causal_reference_is_satisfied_by_stored_op():
    sid = create({"count": 0}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/count", "set", 1)])
    # bob:1 references alice:1 delivered in an earlier batch
    r = submit(
        sid,
        1,
        [op("bob", 1, {"alice": 1, "bob": 1}, "/count", "set", 2)],
    )
    assert r.status_code == 200
    body = view(sid).json()
    assert body["document"]["count"] == 2
    assert body["version"] == {"alice": 1, "bob": 1}
    assert body["conflicts"] == []  # causally later, not concurrent


def test_recomputed_tombstones_go_away_when_overridden():
    sid = create({"a": 1}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/a", "delete")])
    assert view(sid).json()["tombstones"] == ["/a"]
    submit(sid, 1, [op("alice", 2, {"alice": 2}, "/a", "set", 42)])
    body = view(sid).json()
    assert body["document"] == {"a": 42}
    assert body["tombstones"] == []


def test_accumulated_log_equals_single_merge():
    from app.engine import merge_logs

    document = {"items": ["x"]}
    sid = create(document).json()["session_id"]
    batch_a = [
        op("alice", 1, {"alice": 1}, "/items", "insert", "A", index=0),
        op("alice", 2, {"alice": 2}, "/y", "set", 1),
    ]
    batch_b = [op("bob", 1, {"bob": 1}, "/items", "insert", "B", index=0)]
    submit(sid, 0, batch_a)
    submit(sid, 1, batch_b)
    body = view(sid).json()
    single = merge_logs(document, {}, [[Operation.model_validate(o) for o in batch_a + batch_b]])
    assert body["document"] == single["document"]
    assert body["version"] == single["version"]
    assert body["tombstones"] == single["tombstones"]
    assert body["conflicts"] == single["conflicts"]


# ---------------------------------------------------------------------------
# 422: same id different payload, illegal log, execution failure (no partial)
# ---------------------------------------------------------------------------


def test_same_id_different_payload_in_batch_422_and_no_partial_write():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    r = submit(
        sid,
        1,
        [
            op("bob", 1, {"bob": 1}, "/z", "set", "kept?"),
            op("alice", 1, {"alice": 1}, "/x", "set", 2),
        ],
    )
    assert r.status_code == 422
    assert "different payloads" in r.json()["detail"]
    # revision untouched and bob:1 was not partially stored
    body = view(sid).json()
    assert body["revision"] == 1
    assert body["document"] == {"x": 1}


def test_same_id_different_payload_within_batch_422():
    sid = create({}).json()["session_id"]
    r = submit(
        sid,
        0,
        [
            op("alice", 1, {"alice": 1}, "/x", "set", 1),
            op("alice", 1, {"alice": 1}, "/x", "set", 2),
        ],
    )
    assert r.status_code == 422
    assert view(sid).json()["revision"] == 0


def test_accumulated_log_illegal_dependency_422_and_no_partial_write():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    r = submit(
        sid,
        1,
        [
            op("bob", 1, {"bob": 1}, "/z", "set", "kept?"),
            op("bob", 2, {"alice": 9, "bob": 2}, "/w", "set", 0),
        ],
    )
    assert r.status_code == 422
    assert "depends on alice:9" in r.json()["detail"]
    body = view(sid).json()
    assert body["revision"] == 1
    assert "z" not in body["document"]


def test_sequence_gap_against_initial_version_422():
    sid = create({}, {"alice": 2}).json()["session_id"]
    r = submit(sid, 0, [op("alice", 4, {"alice": 4}, "/x", "set", 1)])
    assert r.status_code == 422
    assert "not contiguous" in r.json()["detail"]


def test_execution_failure_422_and_no_partial_write():
    sid = create({"items": [1, 2]}).json()["session_id"]
    r = submit(
        sid,
        0,
        [
            op("alice", 1, {"alice": 1}, "/other", "insert", "x", index=0),
            op("alice", 2, {"alice": 2}, "/items/9", "set", "bad"),
        ],
    )
    assert r.status_code == 422
    body = view(sid).json()
    assert body["revision"] == 0
    assert body["document"] == {"items": [1, 2]}


# ---------------------------------------------------------------------------
# 409: stale expected revision
# ---------------------------------------------------------------------------


def test_stale_revision_409_and_no_write():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    r = submit(sid, 0, [op("bob", 1, {"bob": 1}, "/y", "set", 2)])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "expected revision 0" in detail
    assert "revision 1" in detail
    body = view(sid).json()
    assert body["revision"] == 1
    assert body["document"] == {"x": 1}


def test_future_expected_revision_also_409():
    sid = create({}).json()["session_id"]
    r = submit(sid, 5, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert r.status_code == 409


def test_recovering_after_409_with_current_revision_succeeds():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert submit(sid, 0, [op("bob", 1, {"bob": 1}, "/y", "set", 2)]).status_code == 409
    r = submit(sid, 1, [op("bob", 1, {"bob": 1}, "/y", "set", 2)])
    assert r.status_code == 200
    assert view(sid).json()["document"] == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# atomicity: concurrent commits against the same revision, exactly one wins
# ---------------------------------------------------------------------------


def test_concurrent_commits_against_same_revision_exactly_one_wins():
    sid = create({}).json()["session_id"]
    barrier = threading.Barrier(2)
    results: list[int] = []
    results_lock = threading.Lock()

    def worker(actor):
        s = store_mod.get_store()
        barrier.wait()
        try:
            s.append_batch(
                sid,
                0,
                [opm(actor, 1, {actor: 1}, f"/{actor}", "set", True)],
            )
            with results_lock:
                results.append(200)
        except store_mod.StaleRevision:
            with results_lock:
                results.append(409)

    threads = [threading.Thread(target=worker, args=(a,)) for a in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == [200, 409]
    body = view(sid).json()
    assert body["revision"] == 1  # exactly one batch landed
    assert len(body["document"]) == 1


def test_concurrent_commits_at_distinct_revisions_serialize():
    sid = create({}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    barrier = threading.Barrier(2)
    statuses = {}
    lock = threading.Lock()

    def worker(actor):
        s = store_mod.get_store()
        # alice continues at seq 2; bob issues his first op; both saw rev 1
        batch = (
            [opm("alice", 2, {"alice": 2}, "/alice", "set", True)]
            if actor == "alice"
            else [opm("bob", 1, {"alice": 1, "bob": 1}, "/bob", "set", True)]
        )
        barrier.wait()
        try:
            s.append_batch(sid, 1, batch)
            with lock:
                statuses[actor] = 200
        except store_mod.StaleRevision:
            with lock:
                statuses[actor] = 409

    threads = [threading.Thread(target=worker, args=(a,)) for a in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(statuses.values()) == [200, 409]

    # the loser retries at the new revision and lands as the next revision
    loser = next(a for a, code in statuses.items() if code == 409)
    retry_batch = (
        [op("alice", 2, {"alice": 2}, "/alice", "set", True)]
        if loser == "alice"
        else [op("bob", 1, {"alice": 1, "bob": 1}, "/bob", "set", True)]
    )
    body = view(sid).json()
    assert body["revision"] == 2
    r = submit(sid, 2, retry_batch)
    assert r.status_code == 200
    assert r.json()["revision"] == 3
    body = view(sid).json()
    assert body["version"] == {"alice": 2, "bob": 1}
    assert set(body["document"]) == {"x", "alice", "bob"}


# ---------------------------------------------------------------------------
# durability: sessions survive a process restart (store re-instantiation)
# ---------------------------------------------------------------------------


def test_session_survives_store_reinstantiation(tmp_path):
    sid = create({"title": "draft"}).json()["session_id"]
    submit(sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "v")])
    # simulate container restart: drop all cached stores and reopen the file
    store_mod._stores.clear()
    reopened = store_mod.SessionStore(str(tmp_path / "sessions.db"))
    state = reopened.get_session(sid)
    assert state["revision"] == 1
    assert state["document"] == {"title": "v"}
    assert state["version"] == {"alice": 1}
    # and batches can continue against the reopened store
    reopened.append_batch(
        sid, 1, [opm("bob", 1, {"bob": 1}, "/other", "set", 2)]
    )
    state = reopened.get_session(sid)
    assert state["revision"] == 2
    assert state["document"] == {"title": "v", "other": 2}


def test_two_sessions_are_independent():
    s1 = create({"a": 1}).json()["session_id"]
    s2 = create({"b": 2}).json()["session_id"]
    submit(s1, 0, [op("alice", 1, {"alice": 1}, "/a", "set", 10)])
    assert view(s1).json()["document"] == {"a": 10}
    assert view(s2).json() == {
        "session_id": s2,
        "revision": 0,
        "document": {"b": 2},
        "version": {},
        "conflicts": [],
        "tombstones": [],
    }
