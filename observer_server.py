
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import json
import asyncio
from datetime import datetime

# ==================== DATA MODELS ====================

class TelemetryData(BaseModel):
    timestamp: float
    vision_b64: Optional[str] = None
    metrics: Dict[str, Any]
    action: Dict[str, Any]
    logs: Optional[List[str]] = []

# ==================== SERVER STATE ====================

app = FastAPI(title="Ultron Observer Node")

class GlobalState:
    latest: Optional[TelemetryData] = None
    history: List[TelemetryData] = []
    
state = GlobalState()

# ==================== ENDPOINTS ====================

@app.get("/")
def home():
    return {"status": "Observer Node Online", "last_update": state.latest.timestamp if state.latest else 0}

@app.post("/update")
async def update_telemetry(data: TelemetryData):
    """Receive live telemetry from Ultron Agent."""
    state.latest = data
    # Keep small history for graphs (last 100 points)
    state.history.append(data)
    if len(state.history) > 100:
        state.history.pop(0)
    return {"status": "ack"}

@app.get("/telemetry")
def get_telemetry():
    """Poll latest state for Dashboard."""
    if not state.latest:
        return {"status": "waiting_for_agent"}
    return state.latest

@app.get("/history")
def get_history():
    """Poll history for graphs."""
    return [
        {
            "timestamp": d.timestamp,
            "entropy": d.metrics.get("consciousness", {}).get("entropy", 0.0),
            "surprise": d.metrics.get("consciousness", {}).get("surprise", 0.0),
            "mode": d.metrics.get("mode", "Sys1")
        }
        for d in state.history
    ]

if __name__ == "__main__":
    # Standard port 8000
    print("Starting Observer Server on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
