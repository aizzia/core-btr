import os
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel
from core_btr import CoreBTRPod

BUNDLE_PATH = os.environ.get("BTR_BUNDLE_PATH", "artifacts/duke_pod.pt")
pod = CoreBTRPod.load(BUNDLE_PATH)

mcp = FastMCP("CoreBTR-Constellation-Pod")


@mcp.tool()
def lookup_historical_twins(entity_id: str, top_k: int = 5) -> dict:
    """
    Finds historical entities/runs matching `entity_id` across un-flattened Set Transformer lenses.
    Returns EMPTY_BELOW_CUTOFF when no authentic historical twin clears the calibrated cutoff.
    """
    return pod.query_by_id(entity_id=entity_id, top_k=top_k)


@mcp.tool()
def match_live_event_window(events: List[Dict[str, Any]], top_k: int = 5) -> dict:
    """
    Evaluates a live sequential window of structured rows across all admitted frozen lenses.
    Separates explicit 0.0 events from NULL gaps and refuses out-of-regime anomalies cleanly.
    """
    return pod.query_live_events(events=events, top_k=top_k)


@mcp.tool()
def diagnose_artiq_run_drift(scan_steps: List[Dict[str, Any]], top_k: int = 5) -> dict:
    """
    Companion MCP historical-memory tool for Duke DAX / artiq-mcp:
    Compares a live ARTIQ trapped-ion scan against historical `rid`s across un-flattened lenses
    (PMT magnitudes, stage/beam roles, explicit 0-photon dark readouts vs NaN, shot cadence,
    and drift trajectory). Returns EMPTY_BELOW_CUTOFF when metacognitive re-framing or human
    escalation is required.
    """
    return pod.query_live_events(events=scan_steps, top_k=top_k)


mcp_app = mcp.http_app(path="/mcp")
app = FastAPI(title="Core BTR Drop-In Pod", lifespan=mcp_app.lifespan)
app.mount("/mcp-server", mcp_app)


class LiveWindowRequest(BaseModel):
    events: List[Dict[str, Any]]
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "status": "READY_FROZEN",
        "indexed_entities": len(pod.db_ids),
        "admitted_lenses": pod.audit.admitted_lenses,
        "max_pairwise_spearman": round(pod.audit.max_pairwise_spearman, 4),
        "tau_cutoff": round(pod.audit.tau_cutoff, 4),
        "lens_sha256": pod.lens_hashes,
    }


@app.get("/similar/{entity_id}")
def similar_by_id(entity_id: str, top_k: int = 5):
    res = pod.query_by_id(entity_id=entity_id, top_k=top_k)
    if res["status"] == "NOT_IN_CORPUS":
        raise HTTPException(status_code=404, detail=res)
    return res


@app.post("/match-window")
def match_window(req: LiveWindowRequest):
    try:
        return pod.query_live_events(events=req.events, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
