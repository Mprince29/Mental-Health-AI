# Mental Health AI — Backend API

A production-ready REST API for mental health text analysis, built on two
fine-tuned LLMs trained entirely locally on Apple Silicon (M4, 16GB).

## What It Does

| Endpoint | Description |
|---|---|
| `POST /classify` | Detects mental health condition from text (14 classes) |
| `POST /support` | Classifies text + generates empathetic response |
| `POST /chat` | Multi-turn conversation with session memory |
| `POST /batch-classify` | Classify up to 20 texts in one request |
| `GET /history/{id}` | Retrieve conversation history |
| `DELETE /history/{id}` | Clear conversation history |
| `GET /docs` | Interactive Swagger UI |

## Models

Both models are Qwen2.5-3B-Instruct fine-tuned with LoRA via MLX-LM:

- **Classifier** — 14-class mental health condition detector
  - Trained on 12,880 examples across 14 conditions
  - Source: Reddit mental health communities (~65,000 raw posts), cleaned and balanced
  - Final val loss: 1.572

- **Chatbot** — Empathetic support response generator
  - Trained on 2,352 examples
  - Training data: 2,940 synthetic (post → response) pairs generated locally via Ollama + llama3.2:3b
  - Final val loss: 2.003

## Training Pipeline

```
Raw CSVs (~65,000 posts)
  → scripts/01_audit.py          audit class distribution
  → scripts/02_prepare_data.py   clean + balance → 16,100 examples
  → scripts/03_generate_replies.py  generate replies via Ollama/llama3.2
  → scripts/04_build_chatbot_dataset.py  format for fine-tuning
  → mlx_lm.lora (classifier)    fine-tune classifier
  → mlx_lm.lora (chatbot)       fine-tune chatbot
  → mlx_lm.fuse                 fuse adapters → standalone models
  → uvicorn src.app:app          serve via FastAPI
```

## Project Structure

```
mental-health-ai/
├── src/
│   ├── app.py        # FastAPI app — routes, middleware, ML inference
│   ├── config.py     # All settings (model paths, temps, limits)
│   ├── models.py     # Pydantic request/response schemas
│   └── logger.py     # Structured logging to console + daily log file
├── scripts/
│   ├── 01_audit.py               explore raw data, class counts
│   ├── 02_prepare_data.py        clean + balance classifier dataset
│   ├── 03_generate_replies.py    generate chatbot training replies via Ollama
│   └── 04_build_chatbot_dataset.py  format reply pairs as JSONL
├── configs/
│   ├── lora_classifier.yaml      MLX-LM LoRA config for classifier
│   └── lora_chatbot.yaml         MLX-LM LoRA config for chatbot
├── data_classifier/              Classifier training data (JSONL)
├── data_chatbot/                 Chatbot training data (JSONL)
├── requirements.txt
├── .env.example                  Environment variable reference
└── README.md
```

> **Not in repo (too large):** `fused_classifier/`, `fused_chatbot/`, raw CSVs, LoRA adapters

## Setup

```bash
git clone <repo>
cd mental-health-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit environment config
cp .env.example .env

# Models are not included — you need to train them or obtain the weights
# See Training Pipeline above

uvicorn src.app:app --reload --port 8000
```

## Environment Variables

```
CLASSIFIER_MODEL=./fused_classifier
CHATBOT_MODEL=./fused_chatbot
CLASSIFIER_TEMP=0.1
CHATBOT_TEMP=0.8
CHATBOT_MAX_TOKENS=150
MAX_HISTORY_TURNS=5
HOST=0.0.0.0
PORT=8000
```

## Tech Stack

| Tool | Purpose |
|---|---|
| MLX-LM | LoRA fine-tuning on Apple Silicon |
| Qwen2.5-3B-Instruct | Base model |
| Ollama + llama3.2:3b | Synthetic training data generation |
| FastAPI | REST API |
| Pydantic v2 | Request/response validation |
| uvicorn | ASGI server |

## Condition Classes

`ADHD` `Addiction` `Anxiety` `Bipolar` `Depression` `EatingDisorder`
`Loneliness` `Normal` `OCD` `PTSD` `PersonalityDisorder`
`Schizophrenia` `Stress` `Suicidal`.
