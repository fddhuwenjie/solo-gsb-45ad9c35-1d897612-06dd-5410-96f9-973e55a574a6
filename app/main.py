"""FastAPI application exposing the offline JSON merge service."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .engine import LogValidationError, merge_logs
from .models import (
    ApplyRequest,
    BatchSubmitRequest,
    BatchSubmitResponse,
    MergeRequest,
    MergeResult,
    OperationId,
    QueryRequest,
    QueryResponse,
    ResolveRequest,
    ResolveResult,
    RouteChangesRequest,
    RouteChangesResult,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionView,
)
from .query import QueryValidationError, run_queries
from .resolve import ResolveValidationError, resolve_documents
from .routing import RoutingValidationError, route_changes
from .store import SessionNotFound, StaleRevision, get_store

app = FastAPI(
    title="Offline JSON Merge API",
    version="1.5.0",
    description=(
        "Concurrent, offline-first merging of JSON document operation logs "
        "using vector clocks, deterministic (actor, seq) arbitration, "
        "tombstones and stable array inserts, with optional pre-merge "
        "operation exclusion and replay-log export; snapshot diffing with "
        "wildcard subscriber routing (POST /route-changes); restricted "
        "JSONPath querying (POST /query); and cross-document reference "
        "resolution with overlays (POST /resolve)."
    ),
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _excluded_ids(exclude: list[OperationId] | None) -> list[tuple[str, int]] | None:
    if not exclude:
        return None
    return [(item.actor, item.seq) for item in exclude]


def _merge_response(result: dict) -> JSONResponse:
    """Serialize a merge result, omitting pruning fields absent by request."""
    payload = MergeResult.model_validate(result).model_dump(mode="json")
    if result.get("replay_log") is None:
        payload.pop("replay_log", None)
        payload.pop("excluded", None)
    return JSONResponse(payload)


@app.post("/merge", response_model=MergeResult)
def merge(req: MergeRequest) -> JSONResponse:
    """Merge the operation logs of two replicas of the same initial document."""
    try:
        result = merge_logs(
            req.document,
            req.version,
            [req.replica_a, req.replica_b],
            exclude=_excluded_ids(req.exclude),
        )
    except LogValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _merge_response(result)


@app.post("/apply", response_model=MergeResult)
def apply(req: ApplyRequest) -> JSONResponse:
    """Replay a single replica's operation log onto a document."""
    try:
        result = merge_logs(
            req.document,
            req.version,
            [req.operations],
            exclude=_excluded_ids(req.exclude),
        )
    except LogValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _merge_response(result)


@app.post(
    "/route-changes",
    response_model=RouteChangesResult,
)
def route_changes_endpoint(req: RouteChangesRequest) -> dict:
    """Diff two JSON snapshots and route change events to subscribers."""
    try:
        return route_changes(req.before, req.after, req.subscriptions)
    except RoutingValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> dict:
    """Evaluate restricted JSONPath queries against a document (read-only)."""
    try:
        return run_queries(req.document, req.queries)
    except QueryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/resolve", response_model=ResolveResult)
def resolve_endpoint(req: ResolveRequest) -> dict:
    """Materialize cross-document JSON references into a single document."""
    try:
        return resolve_documents(req.documents, req.root)
    except ResolveValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# durable offline sessions: batched operation-log delivery
# ---------------------------------------------------------------------------


@app.post("/sessions", response_model=SessionCreateResponse, status_code=201)
def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    """Create a session from a common initial document; revision starts at 0."""
    try:
        session_id, revision = get_store().create_session(
            req.document, req.version
        )
    except LogValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SessionCreateResponse(session_id=session_id, revision=revision)


@app.post(
    "/sessions/{session_id}/batches", response_model=BatchSubmitResponse
)
def submit_batch(
    session_id: str, req: BatchSubmitRequest
) -> BatchSubmitResponse:
    """Append one non-empty batch against the expected current revision."""
    try:
        outcome = get_store().append_batch(
            session_id, req.expected_revision, req.operations
        )
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    except StaleRevision as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LogValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchSubmitResponse(session_id=session_id, **outcome)


@app.get("/sessions/{session_id}", response_model=SessionView)
def read_session(session_id: str) -> SessionView:
    """Read revision, document, version vector, conflicts and tombstones."""
    try:
        state = get_store().get_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    return SessionView.model_validate(state)
