"""End-to-end tests for durable batched-submission sessions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import RevisionConflictError, SessionStore


@pytest.fixture()
def client(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
    return TestClient(app)


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


def create_session(client, document=None, version=None):
    resp = client.post(
        "/sessions", json={"document": document, "version": version or {}}
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def submit(client, session_id, expected_revision, operations):
    return client.post(
        f"/sessions/{session_id}/batches",
        json={"expected_revision": expected_revision, "operations": operations},
    )


# ---------------------------------------------------------------------------
# creation and reading
# ---------------------------------------------------------------------------

def test_create_session_returns_revision_zero(client):
    resp = client.post(
        "/sessions", json={"document": {"title": "draft"}, "version": {}}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 0
    assert body["session_id"]


def test_new_session_reflects_initial_document_and_version(client):
    sid = create_session(client, {"a": 1}, {"carol": 2})
    resp = client.get(f"/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["revision"] == 0
    assert body["document"] == {"a": 1}
    assert body["version"] == {"carol": 2}
    assert body["conflicts"] == []
    assert body["tombstones"] == []


def test_negative_initial_version_rejected(client):
    resp = client.post("/sessions", json={"document": {}, "version": {"a": -1}})
    assert resp.status_code == 422


def test_unknown_session_returns_404(client):
    missing = "0" * 32
    assert client.get(f"/sessions/{missing}").status_code == 404
    resp = submit(client, missing, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# batch commits
# ---------------------------------------------------------------------------

def test_batch_commit_increments_revision(client):
    sid = create_session(client, {"count": 0})
    resp = submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/count", "set", 1)])
    assert resp.status_code == 200
    assert resp.json() == {
        "session_id": sid,
        "revision": 1,
        "accepted": 1,
        "duplicates": 0,
    }
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"count": 1}
    assert state["version"] == {"alice": 1}


def test_cross_batch_causal_reference_resolves(client):
    # bob's operation in batch 2 causally references alice:1 stored by batch 1
    sid = create_session(client, {"count": 0})
    r1 = submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/count", "set", 1)])
    assert r1.status_code == 200
    r2 = submit(
        client,
        sid,
        1,
        [op("bob", 1, {"alice": 1, "bob": 1}, "/count", "set", 2)],
    )
    assert r2.status_code == 200
    assert r2.json()["revision"] == 2
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"count": 2}
    assert state["version"] == {"alice": 1, "bob": 1}
    assert state["conflicts"] == []  # causally ordered, not concurrent


def test_late_concurrent_write_joins_original_arbitration(client):
    # bob's write arrives in a later batch but is concurrent with alice:1;
    # recomputing from the initial document keeps it in the original
    # arbitration and conflict statistics
    sid = create_session(client, {"title": "draft"})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "from-alice")])
    resp = submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/title", "set", "from-bob")])
    assert resp.status_code == 200
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"title": "from-bob"}  # ("bob",1) wins
    assert len(state["conflicts"]) == 1
    conflict = state["conflicts"][0]
    assert conflict["path"] == "/title"
    assert conflict["winner"]["actor"] == "bob"
    assert [l["actor"] for l in conflict["losers"]] == ["alice"]


def test_tombstones_visible_in_session_state(client):
    sid = create_session(client, {"title": "draft", "keep": 1})
    submit(client, sid, 0, [op("zoe", 1, {"zoe": 1}, "/title", "delete")])
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"keep": 1}
    assert state["tombstones"] == ["/title"]


def test_concurrent_inserts_across_batches_keep_stable_order(client):
    sid = create_session(client, {"items": ["x"]})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/items", "insert", "A", index=0)])
    submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/items", "insert", "B", index=0)])
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"]["items"] == ["A", "B", "x"]


# ---------------------------------------------------------------------------
# retransmission and duplicates
# ---------------------------------------------------------------------------

def test_retransmitted_batch_does_not_increment_revision(client):
    batch = [
        op("alice", 1, {"alice": 1}, "/x", "set", 1),
        op("alice", 2, {"alice": 2}, "/y", "set", 2),
    ]
    sid = create_session(client, {})
    r1 = submit(client, sid, 0, batch)
    assert r1.json()["revision"] == 1
    r2 = submit(client, sid, 1, batch)  # exact retransmission
    assert r2.status_code == 200
    assert r2.json() == {
        "session_id": sid,
        "revision": 1,  # unchanged: no new operations
        "accepted": 0,
        "duplicates": 2,
    }
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1, "y": 2}
    assert state["version"] == {"alice": 2}


def test_partial_retransmission_stores_only_new_operations(client):
    sid = create_session(client, {})
    submit(
        client,
        sid,
        0,
        [
            op("alice", 1, {"alice": 1}, "/x", "set", 1),
            op("alice", 2, {"alice": 2}, "/y", "set", 2),
        ],
    )
    resp = submit(
        client,
        sid,
        1,
        [
            op("alice", 2, {"alice": 2}, "/y", "set", 2),  # duplicate
            op("alice", 3, {"alice": 3}, "/z", "set", 3),  # new
        ],
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2
    assert resp.json()["accepted"] == 1
    assert resp.json()["duplicates"] == 1
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"x": 1, "y": 2, "z": 3}
    assert state["version"] == {"alice": 3}


def test_same_id_different_payload_rejected_without_writing(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = submit(client, sid, 1, [op("alice", 1, {"alice": 1}, "/x", "set", 2)])
    assert resp.status_code == 422
    assert "different payloads" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1}


# ---------------------------------------------------------------------------
# validation / execution failures -> 422, nothing written
# ---------------------------------------------------------------------------

def test_invalid_batch_leaves_session_untouched(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    # alice:3 without alice:2 -> per-actor log not contiguous
    resp = submit(client, sid, 1, [op("alice", 3, {"alice": 3}, "/y", "set", 2)])
    assert resp.status_code == 422
    assert "not contiguous" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1}
    assert state["version"] == {"alice": 1}


def test_missing_cross_batch_dependency_rejected(client):
    sid = create_session(client, {})
    resp = submit(
        client,
        sid,
        0,
        [op("bob", 1, {"alice": 5, "bob": 1}, "/x", "set", 1)],
    )
    assert resp.status_code == 422
    assert "depends on alice:5" in resp.json()["detail"]
    assert client.get(f"/sessions/{sid}").json()["revision"] == 0


def test_execution_failure_leaves_session_untouched(client):
    sid = create_session(client, {"items": 5})  # not an array
    resp = submit(
        client, sid, 0, [op("alice", 1, {"alice": 1}, "/items", "insert", "A")]
    )
    assert resp.status_code == 422
    assert "not an array" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 0
    assert state["document"] == {"items": 5}


def test_empty_batch_rejected(client):
    sid = create_session(client, {})
    resp = submit(client, sid, 0, [])
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# optimistic concurrency
# ---------------------------------------------------------------------------

def test_stale_expected_revision_returns_409(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = submit(client, sid, 0, [op("bob", 1, {"bob": 1}, "/y", "set", 2)])
    assert resp.status_code == 409
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1}


def test_future_expected_revision_returns_409(client):
    sid = create_session(client, {})
    resp = submit(client, sid, 3, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert resp.status_code == 409
    assert client.get(f"/sessions/{sid}").json()["revision"] == 0


def test_concurrent_same_revision_only_one_commits(tmp_path):
    # validation + recompute + revision commit are atomic: racing batches
    # carrying the same expected revision cannot both succeed
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({}, {})

    def attempt(i):
        try:
            operation = Operation(
                actor=f"actor{i}",
                seq=1,
                version={f"actor{i}": 1},
                path=f"/k{i}",
                action="set",
                value=i,
            )
            return store.submit_batch(rec.session_id, 0, [operation])
        except RevisionConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(16)))

    committed = [o for o in outcomes if o is not None]
    assert len(committed) == 1
    assert committed[0].revision == 1
    assert store.get(rec.session_id).revision == 1


# ---------------------------------------------------------------------------
# durability
# ---------------------------------------------------------------------------

def test_session_survives_store_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_session(client, {"count": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/count", "set", 1)])
    submit(
        client,
        sid,
        1,
        [op("bob", 1, {"alice": 1, "bob": 1}, "/count", "set", 2)],
    )

    # simulate a service restart: a brand-new store over the same directory
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 2
    assert state["document"] == {"count": 2}
    assert state["version"] == {"alice": 1, "bob": 1}

    # and the log is still enforced against the persisted state
    resp = submit(client, sid, 2, [op("alice", 3, {"alice": 3}, "/count", "set", 3)])
    assert resp.status_code == 422  # alice:2 was never submitted
    resp = submit(client, sid, 0, [op("carol", 1, {"carol": 1}, "/x", "set", 1)])
    assert resp.status_code == 409  # revision survived the restart too
