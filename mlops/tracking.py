"""
Layer: MLOPS
Purpose: MLflow experiment tracking — wraps all training and eval runs.
"""
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_tracker(tracking_uri: str = "mlruns", experiment_name: str = "eurosec-ai"):
    """Returns configured MLflow client."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed — tracking disabled")
        return None


def track_run(run_name: str, tags: dict = None):
    """Decorator to wrap any function call in an MLflow run."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                import mlflow
                with mlflow.start_run(run_name=run_name, tags=tags or {}):
                    return fn(*args, **kwargs)
            except ImportError:
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def log_fine_tune_run(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    epochs: int,
    train_loss: float,
    eval_loss: float,
    ragas_faithfulness: float,
    ragas_relevancy: float,
    pii_f1: float,
    adversarial_pass_rate: float,
    artifact_path: str = None,
) -> None:
    """Log a complete fine-tuning run with all relevant metrics."""
    mlflow = get_tracker()
    if not mlflow:
        return

    with mlflow.start_run(run_name=f"finetune_{model_name}"):
        mlflow.log_params({
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "epochs": epochs,
        })
        mlflow.log_metrics({
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "ragas_faithfulness": ragas_faithfulness,
            "ragas_relevancy": ragas_relevancy,
            "pii_f1": pii_f1,
            "adversarial_pass_rate": adversarial_pass_rate,
        })
        if artifact_path:
            mlflow.log_artifact(artifact_path)
        logger.info("Fine-tune run logged to MLflow")
