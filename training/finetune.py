"""
Layer: TRAINING (offline pipeline)
Purpose: QLoRA fine-tuning of Mistral 7B on GDPR/legal domain data.
         Runs on Google Colab T4 or any GPU with 4GB+ VRAM.
         After training: merge weights → GGUF → load into Ollama.

Usage:
    python training/finetune.py --data data/training/gdpr_qa.jsonl
"""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "output_dir": "models/mistral-eurosec-lora",
}


def load_dataset(data_path: str):
    """Load JSONL training data in Alpaca format."""
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            # Alpaca format: instruction, input (optional), output
            prompt = f"### Instruction:\n{item['instruction']}\n"
            if item.get("input"):
                prompt += f"\n### Input:\n{item['input']}\n"
            prompt += f"\n### Response:\n{item['output']}"
            data.append({"text": prompt})
    logger.info("Loaded %d training examples", len(data))
    return data


def run_finetuning(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    data_path: str = "data/training/gdpr_qa.jsonl",
    output_dir: str = "models/mistral-eurosec-lora",
) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset
        import mlflow
    except ImportError as e:
        raise RuntimeError(f"Training dependencies missing: {e}. Run: pip install transformers peft trl datasets bitsandbytes") from e

    mlflow.set_experiment("eurosec-finetuning")

    with mlflow.start_run(run_name=f"qlora_{base_model.split('/')[-1]}"):
        mlflow.log_params({**QLORA_CONFIG, **TRAINING_CONFIG, "base_model": base_model})

        # ── Load model in 4-bit ────────────────────────────────────────────
        logger.info("Loading %s in 4-bit quantization...", base_model)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token

        # ── Apply QLoRA ────────────────────────────────────────────────────
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(**QLORA_CONFIG)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Load data ──────────────────────────────────────────────────────
        raw_data = load_dataset(data_path)
        dataset = Dataset.from_list(raw_data)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        # ── Train ──────────────────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=output_dir,
            report_to="mlflow",
            **TRAINING_CONFIG,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            dataset_text_field="text",
            max_seq_length=1024,
            args=training_args,
        )
        trainer.train()

        # ── Save adapter ───────────────────────────────────────────────────
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(output_dir)
        logger.info("LoRA adapter saved to %s", output_dir)
        logger.info("Next step: run training/export.py to merge weights and create GGUF")


def main():
    parser = argparse.ArgumentParser(description="EuroSec AI QLoRA fine-tuning")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data", default="data/training/gdpr_qa.jsonl")
    parser.add_argument("--output", default="models/mistral-eurosec-lora")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_finetuning(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
