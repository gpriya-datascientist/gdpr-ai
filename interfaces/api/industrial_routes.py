"""
Layer: INTERFACES — Industrial API Routes
Purpose: FastAPI routes for industrial fusion engine and sensor data.
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from industrial.fusion.engine import LocalFusionEngine, SensorReading

logger = logging.getLogger(__name__)
industrial_router = APIRouter()
_fusion_engine = LocalFusionEngine()


class SensorPayload(BaseModel):
    left_door_stroke_rate: float = 45.0
    right_door_stroke_rate: float = 44.0
    hood_cycle_count: int = 12
    assembly_speed_ms: float = 2.8
    torque_nm: float = 150.0
    weld_quality_score: float = 0.95
    cycle_time_seconds: float = 240.0
    temperature_celsius: float = 22.0
    vibration_hz: float = 50.0
    defect_flag: int = 0


class FusionResponse(BaseModel):
    timestamp: str
    anomalies_detected: list[str]
    anomaly_score: float
    failure_risk_48h: float
    speed_optimization_pct: float
    bottleneck: Optional[str]
    recommendation: str
    llm_insight: str


@industrial_router.post("/sensor/analyze", response_model=FusionResponse)
async def analyze_sensor(payload: SensorPayload):
    """Analyze a sensor reading through the local fusion engine."""
    reading = SensorReading(
        timestamp=datetime.utcnow(),
        left_door_stroke_rate=payload.left_door_stroke_rate,
        right_door_stroke_rate=payload.right_door_stroke_rate,
        hood_cycle_count=payload.hood_cycle_count,
        assembly_speed_ms=payload.assembly_speed_ms,
        torque_nm=payload.torque_nm,
        weld_quality_score=payload.weld_quality_score,
        cycle_time_seconds=payload.cycle_time_seconds,
        temperature_celsius=payload.temperature_celsius,
        vibration_hz=payload.vibration_hz,
        defect_flag=payload.defect_flag,
    )
    result = _fusion_engine.analyze(reading)
    return FusionResponse(
        timestamp=result.timestamp.isoformat(),
        anomalies_detected=result.anomalies_detected,
        anomaly_score=result.anomaly_score,
        failure_risk_48h=result.failure_risk_48h,
        speed_optimization_pct=result.speed_optimization_pct,
        bottleneck=result.bottleneck,
        recommendation=result.recommendation,
        llm_insight=result.llm_insight,
    )


@industrial_router.get("/sensor/baselines")
async def get_baselines():
    """Return the BMW assembly line baseline thresholds."""
    from industrial.fusion.engine import BASELINES
    return {"baselines": BASELINES, "speed_target_improvement_pct": 50}


@industrial_router.get("/sensor/health")
async def industrial_health():
    return {
        "status": "ok",
        "fusion_engine": "active",
        "history_size": len(_fusion_engine._history),
    }
