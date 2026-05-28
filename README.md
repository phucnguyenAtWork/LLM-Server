# FINA — Financial Intelligence & Natural-language Assistant

Fine-tuned Qwen 2.5 3B LoRA assistant for personal finance Q&A, grounded in live transactions served by the companion AWAD2 backend.

## Companion repo

FINA reads user transactions, budgets, and category aggregates from **AWAD2** (separate repository) on branch `llm-agent-feature`. Start AWAD2 first; FINA calls `http://<awad2-host>:4001/api/fina/*` for grounded context.

## Prerequisites

- Python 3.12 + `venv`
- **Git LFS** installed before cloning (`git lfs install`) so the v8 LoRA adapter pulls automatically
- MySQL reachable by AWAD2 (FINA only talks to AWAD2, not to MySQL directly except as a fallback)
- NVIDIA GPU with ~6 GB VRAM for 8-bit inference (tested on RTX 5060 Ti, ~2 tok/s)

## Clone

```bash
git lfs install                        # one-time
git clone https://github.com/phucnguyenAtWork/LLM-Server.git
cd LLM-Server
```

After clone, verify the adapter materialised (should be ~115 MB, not a 130-byte pointer):

```bash
ls -lh financial_qwen_native_v8/adapter_model.safetensors
```

If Git LFS is unavailable, download the adapter from the [v1.0-thesis release](https://github.com/phucnguyenAtWork/LLM-Server/releases/tag/v1.0-thesis) and drop it into `financial_qwen_native_v8/`.

## Configure

Copy `.env.example` to `.env` and fill in the database credentials (used by AWAD2-side fallback paths and the LSTM forecasting module).

### `MAC_API_URL` is optional

`MAC_API_URL` controls where FINA reaches AWAD2. The default `http://localhost:4001/api` works when FINA and AWAD2 run on the **same machine** — you do not need to set this variable in that case.

Only override it when running the two services on different hosts:

```bash
# in .env, e.g. when AWAD2 lives on another machine in your LAN / Tailnet
MAC_API_URL=http://<awad2-host>:4001/api
```

## Run

```bash
# 1. start AWAD2 (in the AWAD2 repo, branch llm-agent-feature)
npm run dev

# 2. start FINA (this repo)
python api.py
```

FINA listens on `0.0.0.0:8105` by default. Health check: `GET http://localhost:8105/health`.

## Smoke tests

```bash
python test_console.py     # interactive CLI against the running API
python benchmark.py        # full evaluation grid
```

## Reproducing the thesis numbers

```bash
python run_thesis_matrix.py
```

Runs the 3×3 model × RAG-backend matrix ({baseline, v8} × {none, oracle, vector}) and prints the appendix table. Figures used in the thesis live under `figures/chapter4/` and `figures/defense/`. Train/val/test split IDs are frozen in `splits/`.

## Layout

- `api.py` — FastAPI service (chat, schema, numeric guard, RAG glue)
- `chat.py` / `test_console.py` — CLI front-ends
- `mac_client.py` — AWAD2 HTTP client
- `rag/` — MiniLM + ChromaDB retrieval pipeline
- `forecasting/` — LSTM spend-forecast module
- `train.py` — LoRA fine-tuning entry point
- `benchmark.py`, `thesis_evaluation.py`, `run_thesis_matrix.py` — evaluation harnesses
- `financial_qwen_native_v8/` — frozen LoRA adapter (LFS-tracked)
