"""
Layer: INDUSTRIAL — Fusion Engine
Purpose: Combines live sensor data + LLM reasoning for intelligent factory insights.
         Detects anomalies, predicts failures 48h ahead, recommends optimizations.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    timestamp: datetime
    left_door_stroke_rate: float = 0.0
    right_door_stroke_rate: float = 0.0
    hood_cycle_count: int = 0
    assembly_speed_ms: float = 0.0
    torque_nm: float = 0.0
    weld_quality_score: float = 1.0
    cycle_time_seconds: float = 240.0
    temperature_celsius: float = 22.0
    vibration_hz: float = 50.0
    defect_flag: int = 0


@dataclass
class FusionResult:
    timestamp: datetime
    anomalies_detected: list[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    failure_risk_48h: float = 0.0
    speed_optimization_pct: float = 0.0
    bottleneck: Optional[str] = None
    recommendation: str = ""
    llm_insight: str = ""
    raw_reading: Optional[SensorReading] = None

# Baseline thresholds (BMW assembly line standards)
BASELINES = {
    "left_door_stroke_rate":  {"mean": 45.0, "std": 5.0, "target": 45.0},
    "right_door_stroke_rate": {"mean": 44.0, "std": 5.0, "target": 44.0},
    "assembly_speed_ms":      {"mean": 2.8,  "std": 0.3, "min_target": 4.2},  # +50%
    "cycle_time_seconds":     {"mean": 240,  "std": 20,  "max_target": 160},  # -33%
    "weld_quality_score":     {"mean": 0.95, "std": 0.03, "min_acceptable": 0.85},
    "torque_nm":              {"mean": 150,  "std": 15,  "range": (80, 220)},
    "vibration_hz":           {"mean": 50,   "std": 8,   "max_safe": 90},
}

ANOMALY_Z = 2.5


class LocalFusionEngine:
    """
    Fuses sensor readings with LLM reasoning.
    Step 1: Statistical anomaly detection (z-score + threshold rules)
    Step 2: Failure risk scoring (rolling window trend)
    Step 3: Speed optimization gap analysis
    Step 4: LLM contextual insight generation
    """

    def __init__(self, llm_client=None, history_size: int = 100) -> None:
        self._llm = llm_client
        self._history: list[SensorReading] = []
        self._history_size = history_size

    def analyze(self, reading: SensorReading) -> FusionResult:
        self._history.append(reading)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        result = FusionResult(timestamp=reading.timestamp, raw_reading=reading)

        # Step 1: Anomaly detection
        result.anomalies_detected, result.anomaly_score = self._detect_anomalies(reading)

        # Step 2: Failure risk prediction
        result.failure_risk_48h = self._predict_failure_risk()

        # Step 3: Speed optimization gap
        result.speed_optimization_pct, result.bottleneck = self._speed_gap_analysis(reading)

        # Step 4: Generate recommendation
        result.recommendation = self._generate_recommendation(result)

        # Step 5: LLM insight (if available)
        if self._llm:
            result.llm_insight = self._get_llm_insight(reading, result)

        logger.info("Fusion result: anomalies=%d risk=%.2f speed_gap=%.1f%%",
                    len(result.anomalies_detected), result.failure_risk_48h,
                    result.speed_optimization_pct)
        return result

    def _detect_anomalies(self, r: SensorReading) -> tuple[list[str], float]:
        anomalies = []
        total_z = 0.0
        checks = [
            ("left_door_stroke_rate",  r.left_door_stroke_rate),
            ("right_door_stroke_rate", r.right_door_stroke_rate),
            ("assembly_speed_ms",      r.assembly_speed_ms),
            ("cycle_time_seconds",     r.cycle_time_seconds),
            ("torque_nm",              r.torque_nm),
            ("vibration_hz",           r.vibration_hz),
        ]
        for name, value in checks:
            b = BASELINES.get(name)
            if not b:
                continue
            z = abs(value - b["mean"]) / max(b["std"], 0.001)
            total_z += z
            if z > ANOMALY_Z:
                anomalies.append(f"{name}: z={z:.2f} (value={value:.2f})")

        # Rule-based checks
        if r.weld_quality_score < BASELINES["weld_quality_score"]["min_acceptable"]:
            anomalies.append(f"weld_quality_score below threshold: {r.weld_quality_score:.2f}")
        if r.defect_flag:
            anomalies.append("defect_flag raised")
        if abs(r.left_door_stroke_rate - r.right_door_stroke_rate) > 15:
            anomalies.append(f"door asymmetry: L={r.left_door_stroke_rate:.1f} R={r.right_door_stroke_rate:.1f}")

        return anomalies, round(total_z / max(len(checks), 1), 3)

    def _predict_failure_risk(self) -> float:
        """Trend-based failure risk using recent history."""
        if len(self._history) < 10:
            return 0.0
        recent = self._history[-20:]
        defect_rate = sum(r.defect_flag for r in recent) / len(recent)
        avg_anomaly_score = np.mean([
            abs(r.left_door_stroke_rate - BASELINES["left_door_stroke_rate"]["mean"])
            for r in recent
        ]) / BASELINES["left_door_stroke_rate"]["std"]
        weld_degradation = 1.0 - np.mean([r.weld_quality_score for r in recent])
        risk = min(1.0, (defect_rate * 3) + (avg_anomaly_score / 10) + (weld_degradation * 2))
        return round(risk, 3)

    def _speed_gap_analysis(self, r: SensorReading) -> tuple[float, Optional[str]]:
        """Calculate how far current speed is from 50% improvement target."""
        target_speed = BASELINES["assembly_speed_ms"]["min_target"]
        current = r.assembly_speed_ms
        gap_pct = max(0, (target_speed - current) / target_speed * 100)

        bottleneck = None
        if r.cycle_time_seconds > BASELINES["cycle_time_seconds"]["max_target"]:
            bottleneck = "cycle_time"
        elif abs(r.left_door_stroke_rate - r.right_door_stroke_rate) > 10:
            bottleneck = "door_asymmetry"
        elif r.torque_nm > BASELINES["torque_nm"]["range"][1]:
            bottleneck = "torque_overload"
        elif r.vibration_hz > BASELINES["vibration_hz"]["max_safe"]:
            bottleneck = "vibration"

        return round(gap_pct, 1), bottleneck

    def _generate_recommendation(self, result: FusionResult) -> str:
        if not result.anomalies_detected and result.failure_risk_48h < 0.2:
            return "All systems nominal. No action required."
        parts = []
        if result.failure_risk_48h > 0.7:
            parts.append("HIGH RISK: Schedule immediate maintenance inspection.")
        elif result.failure_risk_48h > 0.4:
            parts.append("MEDIUM RISK: Plan maintenance within 24 hours.")
        if result.bottleneck == "cycle_time":
            parts.append("Cycle time exceeds target — check conveyor belt calibration.")
        elif result.bottleneck == "door_asymmetry":
            parts.append("Door stroke asymmetry detected — inspect left door mechanism.")
        elif result.bottleneck == "torque_overload":
            parts.append("Torque overload on assembly arm — reduce load or recalibrate.")
        elif result.bottleneck == "vibration":
            parts.append("Vibration exceeds safe limit — check bearing wear.")
        if result.speed_optimization_pct > 10:
            parts.append(f"Speed {result.speed_optimization_pct:.0f}% below 50% target — review bottleneck.")
        return " ".join(parts) if parts else "Minor anomalies detected. Monitor closely."

    def _get_llm_insight(self, reading: SensorReading, result: FusionResult) -> str:
        """Get natural language insight from LLM about sensor state."""
        prompt = f"""You are an expert BMW assembly line engineer AI.
Current sensor readings:
- Left door stroke rate: {reading.left_door_stroke_rate:.1f} (baseline: 45.0)
- Right door stroke rate: {reading.right_door_stroke_rate:.1f} (baseline: 44.0)
- Assembly speed: {reading.assembly_speed_ms:.2f} m/s (target: 4.2 m/s for 50% improvement)
- Cycle time: {reading.cycle_time_seconds:.0f}s (target: <160s)
- Weld quality: {reading.weld_quality_score:.2f} (min: 0.85)
- Failure risk 48h: {result.failure_risk_48h:.1%}
- Anomalies: {', '.join(result.anomalies_detected) if result.anomalies_detected else 'None'}

Provide a 2-sentence technical insight about the current state and one specific action."""
        try:
            response = self._llm.generate(prompt=prompt, max_tokens=150)
            return response.text
        except Exception as e:
            logger.warning("LLM insight failed: %s", e)
            return result.recommendation
