"""
Layer: INDUSTRIAL PIPELINE
Purpose: Ingest, clean, and feature-engineer BMW/automotive sensor data from Kaggle.
Usage: python industrial/data_pipeline.py --input data/industrial/raw/sensor_data.csv
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = {
    "left_door_stroke_rate": "float",
    "right_door_stroke_rate": "float",
    "hood_cycle_count": "int",
    "assembly_speed_ms": "float",
    "torque_nm": "float",
    "weld_quality_score": "float",
    "cycle_time_seconds": "float",
    "temperature_celsius": "float",
    "vibration_hz": "float",
    "defect_flag": "int",
    "timestamp": "datetime",
}

ANOMALY_Z_THRESHOLD = 2.5
ROLLING_WINDOWS = [5, 15, 60]  # minutes


class IndustrialDataPipeline:
    """
    Full ETL pipeline for automotive sensor data.
    Steps: Load → Clean → Normalize → Feature Engineer → Anomaly Flag → Export
    """

    def __init__(self, output_dir: str = "data/industrial/processed") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_path: str) -> pd.DataFrame:
        logger.info("Starting industrial data pipeline: %s", input_path)
        df = self._load(input_path)
        df = self._clean(df)
        df = self._normalize(df)
        df = self._feature_engineer(df)
        df = self._flag_anomalies(df)
        self._export(df)
        logger.info("Pipeline complete — %d rows processed", len(df))
        return df

    def _load(self, path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, parse_dates=["timestamp"])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows with all nulls
        df = df.dropna(how="all")
        # Fill numeric nulls with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        # Remove physically impossible values
        if "left_door_stroke_rate" in df.columns:
            df = df[df["left_door_stroke_rate"].between(0, 1000)]
        if "assembly_speed_ms" in df.columns:
            df = df[df["assembly_speed_ms"].between(0, 100)]
        if "cycle_time_seconds" in df.columns:
            df = df[df["cycle_time_seconds"].between(1, 3600)]
        logger.info("After cleaning: %d rows", len(df))
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalize all numeric sensor columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["defect_flag", "hood_cycle_count"]
        cols_to_norm = [c for c in numeric_cols if c not in exclude]
        for col in cols_to_norm:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        return df

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling averages, deltas, and derived features."""
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        sensor_cols = ["left_door_stroke_rate", "right_door_stroke_rate",
                       "assembly_speed_ms", "cycle_time_seconds", "torque_nm"]

        for col in sensor_cols:
            if col not in df.columns:
                continue
            # Rolling means
            for w in ROLLING_WINDOWS:
                df[f"{col}_roll{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            # Delta (rate of change)
            df[f"{col}_delta"] = df[col].diff().fillna(0)

        # Door symmetry score (left vs right door performance)
        if "left_door_stroke_rate" in df.columns and "right_door_stroke_rate" in df.columns:
            df["door_symmetry"] = abs(df["left_door_stroke_rate"] - df["right_door_stroke_rate"])

        # Throughput efficiency ratio
        if "cycle_time_seconds" in df.columns and "assembly_speed_ms" in df.columns:
            df["throughput_efficiency"] = df["assembly_speed_ms"] / df["cycle_time_seconds"].replace(0, np.nan)

        return df

    def _flag_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score anomaly detection on key sensor columns."""
        key_cols = ["left_door_stroke_rate", "right_door_stroke_rate",
                    "assembly_speed_ms", "cycle_time_seconds"]
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = False

        for col in key_cols:
            if col not in df.columns:
                continue
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                z = (df[col] - mean) / std
                df["anomaly_score"] += z.abs()
                df["is_anomaly"] |= (z.abs() > ANOMALY_Z_THRESHOLD)

        anomaly_count = df["is_anomaly"].sum()
        logger.info("Anomaly detection: %d anomalies found (%.1f%%)",
                    anomaly_count, 100 * anomaly_count / max(len(df), 1))
        return df

    def _export(self, df: pd.DataFrame) -> None:
        out = self._output_dir / "processed_sensor_data.csv"
        df.to_csv(out, index=False)
        logger.info("Exported processed data to %s", out)


def generate_sample_dataset(output_path: str = "data/industrial/raw/sensor_data.csv",
                             n_rows: int = 10000) -> None:
    """Generate synthetic BMW assembly line data for testing."""
    import random
    from datetime import datetime, timedelta
    np.random.seed(42)
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_rows)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "left_door_stroke_rate": np.random.normal(45, 5, n_rows).clip(0, 100),
        "right_door_stroke_rate": np.random.normal(44, 5, n_rows).clip(0, 100),
        "hood_cycle_count": np.random.poisson(12, n_rows),
        "assembly_speed_ms": np.random.normal(2.8, 0.3, n_rows).clip(0, 10),
        "torque_nm": np.random.normal(150, 15, n_rows).clip(50, 300),
        "weld_quality_score": np.random.beta(8, 2, n_rows),
        "cycle_time_seconds": np.random.normal(240, 20, n_rows).clip(60, 600),
        "temperature_celsius": np.random.normal(22, 3, n_rows),
        "vibration_hz": np.random.normal(50, 8, n_rows).clip(0, 200),
        "defect_flag": np.random.binomial(1, 0.03, n_rows),
    })
    # Inject anomalies for testing
    anomaly_idx = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[anomaly_idx, "left_door_stroke_rate"] *= 2.5
    df.loc[anomaly_idx, "cycle_time_seconds"] *= 1.8

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d rows of synthetic sensor data -> %s", n_rows, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/industrial/raw/sensor_data.csv")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data first")
    args = parser.parse_args()

    if args.generate:
        generate_sample_dataset(args.input)

    pipeline = IndustrialDataPipeline()
    pipeline.run(args.input)
