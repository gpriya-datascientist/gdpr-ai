# EuroSec AI

GDPR-compliant hybrid local-cloud desktop AI assistant.
Sensitive documents stay local. Only sanitized, PII-free queries reach the cloud.

## Architecture

```
User query → Intent classifier → PII sanitizer → Route decision
                                                     ↓           ↓
                                               LOCAL ONLY   CLOUD (sanitized)
                                               Mistral+RAG  Groq / OpenAI
                                                     ↓           ↓
                                              GDPR audit log ← always logged
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set ENCRYPTION_KEY and GROQ_API_KEY

# 3. Start local LLM (Ollama)
ollama pull mistral:7b-instruct-q4_K_M

# 4. Start API server
python -m uvicorn interfaces.api.main:app --host 127.0.0.1 --port 8000

# 5. Run tests
pytest tests/ -v

# 6. Run adversarial suite
python eval/adversarial.py
```

## Fine-tuning (optional, runs offline)

```bash
# On Google Colab T4 (free):
python training/finetune.py --data data/training/gdpr_qa.jsonl

# After training, export to Ollama:
python training/export.py
```

## Project structure

```
eurosec-ai/
├── domain/          ← entities, interfaces, exceptions (no deps)
├── application/     ← orchestrator, classifier, sanitizer, RAG (domain only)
├── infrastructure/  ← Ollama, Groq, ChromaDB, Presidio, SQLite (all concrete)
├── interfaces/      ← FastAPI routes, IPC handlers
├── config/          ← settings, dependency injection container
├── training/        ← QLoRA fine-tuning pipeline (offline)
├── eval/            ← adversarial suite (50 cases), RAGAS eval
├── mlops/           ← MLflow tracking, Evidently drift monitor
└── tests/           ← unit, integration, adversarial
```

## Clean architecture rules

- `domain/` imports nothing from this project
- `application/` imports only `domain/`
- `infrastructure/` imports `domain/` + `application/`
- `interfaces/` imports everything
- `config/container.py` is the ONLY place where concrete classes are instantiated

## Key metrics (thesis targets)

| Metric | Target |
|---|---|
| Adversarial pass rate | ≥ 96% |
| Critical leaks | 0 |
| PII false-negative rate | < 2% |
| Local query latency p95 | < 3s |
| Cloud query latency p95 | < 5s |
