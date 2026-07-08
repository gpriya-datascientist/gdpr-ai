"""
Layer: INDUSTRIAL — Kaggle Dataset Downloader
Purpose: Download BMW/automotive sensor datasets from Kaggle for pipeline + fine-tuning.
Usage: python industrial/kaggle_downloader.py --dataset <dataset-name>
"""
import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

RECOMMENDED_DATASETS = [
    {
        "name": "predictive-maintenance-dataset",
        "kaggle_id": "arnabbiswas1/microsoft-azure-predictive-maintenance",
        "description": "Microsoft Azure predictive maintenance — machine failure data",
        "columns": ["machineID", "datetime", "volt", "rotate", "pressure", "vibration", "failure"],
    },
    {
        "name": "industrial-iot-sensor",
        "kaggle_id": "inIT-OWL/production-plant-data-for-condition-monitoring",
        "description": "Production plant sensor data for condition monitoring",
        "columns": ["timestamp", "sensor_readings", "machine_status"],
    },
    {
        "name": "manufacturing-defects",
        "kaggle_id": "rabieelkharoua/predicting-manufacturing-defects-dataset",
        "description": "Manufacturing defect prediction dataset",
        "columns": ["ProductionVolume", "ProductionCost", "DefectRate", "DefectStatus"],
    },
]


def download_dataset(kaggle_id: str, output_dir: str = "data/industrial/raw") -> str:
    """Download a Kaggle dataset. Requires kaggle API credentials."""
    try:
        import kaggle
    except ImportError:
        raise RuntimeError("Install kaggle: pip install kaggle")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading: %s -> %s", kaggle_id, output_dir)
    kaggle.api.dataset_download_files(kaggle_id, path=str(output_path), unzip=True)
    logger.info("Download complete: %s", output_dir)
    return str(output_path)


def list_recommended() -> None:
    print("\nRecommended datasets for BMW/Industrial fine-tuning:\n")
    for i, ds in enumerate(RECOMMENDED_DATASETS, 1):
        print(f"{i}. {ds['name']}")
        print(f"   Kaggle ID: {ds['kaggle_id']}")
        print(f"   Description: {ds['description']}")
        print(f"   Columns: {', '.join(ds['columns'])}\n")


def setup_kaggle_credentials() -> None:
    """Guide user to set up Kaggle API credentials."""
    print("\nTo download Kaggle datasets:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section → 'Create New API Token'")
    print("3. Save kaggle.json to: C:\\Users\\gpngur01\\.kaggle\\kaggle.json")
    print("4. Run: pip install kaggle")
    print("5. Then run this script again with --dataset flag\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download industrial datasets from Kaggle")
    parser.add_argument("--dataset", help="Kaggle dataset ID", default=None)
    parser.add_argument("--list", action="store_true", help="List recommended datasets")
    parser.add_argument("--setup", action="store_true", help="Show credential setup guide")
    parser.add_argument("--output", default="data/industrial/raw")
    args = parser.parse_args()

    if args.setup or not (args.dataset or args.list):
        setup_kaggle_credentials()
    elif args.list:
        list_recommended()
    elif args.dataset:
        download_dataset(args.dataset, args.output)
