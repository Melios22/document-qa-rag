# Vietnamese PDF RAG — End‑to‑End

Build a recruiter‑ready, end‑to‑end Retrieval‑Augmented Generation (RAG) system for Vietnamese PDF documents: preprocess PDFs → build a hybrid vector DB (dense + sparse) → ask questions via CLI, API, or Streamlit UI with multi‑LLM support (Gemini, Watsonx).

## Features

- Hybrid search with BGE‑M3 (dense + sparse) and Milvus, with RRF fusion and reranking
- Unified class `VietnameseRAG` as a single, clean entry point for the full flow
- Practical packaging: logs, configs, Docker for Milvus, env templating, and runbooks
- Designed for Vietnamese: text cleaning, token heuristics, stopwords, multilingual embeddings

---

## Project structure

```text
document-qa-rag/
│
├─ main_preprocess.py        # Step 1: Process PDF documents
├─ main_build_rag.py         # Step 2: Build vector database
├─ main_search_rag.py        # Step 3: Interactive RAG queries (CLI)
├─ streamlit_app.py          # Web UI interface
├─ simple_api.py             # Minimal API interface
│
├─ src/
│  ├─ preprocess/            # PDF extraction, cleaning, chunk metadata, storage
│  ├─ rag_builder/           # Vector DB (Milvus) building + BGE‑M3 encoder
│  ├─ rag_retriever/         # Unified RAG + hybrid search + LLMs
│  ├─ utils/                 # Logging utilities
│  ├─ config.json            # System configuration
│  └─ constant.py            # Constants loaded from config
│
├─ data/                     # PDFs, processed outputs, local Milvus DB
├─ logs/                     # Preprocess/build/retrieval/error/QA logs
├─ requirements/                     # Requirements per stage
├─ docker-compose.yml        # Optional: Milvus service
├─ .env.example              # Environment template
└─ requirements.txt          # Install all stages at once
```

---

## Quick start

1) Python and environment

- Python 3.10+
- Optional: Docker (for Milvus service)

2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reqs/requirements-preprocess.txt   # stage 1: preprocess
pip install -r reqs/requirements-build.txt        # stage 2: build vector DB
pip install -r reqs/requirements-retrieval.txt    # stage 3: retrieval + UI/API
```

3) Configure environment

Create a `.env` file using `.env.example` as a reference:

```env
# Google Gemini
GEMINI_API_KEY=...

# IBM Watsonx
WATSONX_URL=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
```

4) Run the pipeline

- Preprocess PDFs in `data/pdfs/`:

```bash
python ./main_preprocess.py
```

- Build the hybrid vector DB:

```bash
python ./main_build_rag.py
```

- Query from CLI:

```bash
python ./main_search_rag.py
```

Alternative interfaces:

- Streamlit UI:

```bash
streamlit run ./streamlit_app.py
```

- Minimal API:

```bash
python ./simple_api.py
```

---

## Usage

Put PDFs in `data/pdfs/`, then run:

```bash
# 1) Preprocess
python ./main_preprocess.py

# 2) Build vector DB
python ./main_build_rag.py

# 3) Ask questions (CLI)
python ./main_search_rag.py

# Optional UIs
streamlit run ./streamlit_app.py   # Web UI
python ./simple_api.py             # Minimal API
```

---

## Concepts

- Hybrid search: BGE‑M3 (dense + sparse) + RRF + optional reranker
- Pluggable LLMs: Gemini or Watsonx via a simple factory
- Vietnamese‑aware processing: cleaning, stopwords, UTF‑8 safety

---

## Configuration

Main config: `src/config.json` (key excerpts)

Note: You can change settings in `src/config.json` to match your environment (model IDs, Milvus connection, and search/retrieval parameters like k and rerank_top_k).

```json
{
  "embedding_model": { "model_id": "BAAI/bge-m3" },
  "reranker_model": { "model_id": "BAAI/bge-reranker-v2-m3" },
  "vector_database": {
    "connection": { "use_docker": false, "collection_name": "vndoc_rag_hybrid" },
    "hybrid_search": { "dense_weight": 0.7, "sparse_weight": 0.3, "rrf_k": 30 }
  },
  "search_retrieval": { "vector_search": { "default_k": 10 }, "reranking": { "rerank_top_k": 3 } }
}
```

Environment (`.env`)

```env
GEMINI_API_KEY=...
WATSONX_URL=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
```

---

## Deployment

Local Milvus (Docker):

```bash
docker-compose up -d
```

Production notes

- GPU for faster embedding generation (optional)
- Monitor LLM API usage/quotas
- Consider Milvus cluster for scale + persistence
- Add caching on top of retrieval or answers if needed

---

## Troubleshooting

Common issues

- "No documents found": add PDFs to `data/pdfs/` and rerun preprocessing
- "Vector database not found": build via `main_build_rag.py`
- LLM errors: check API keys in `.env` and network; verify provider quotas

Logs

- `logs/preprocess.log` — document processing
- `logs/builder.log` — vector DB building
- `logs/retriever.log` — RAG retrieval
- `logs/errors.log` — errors
- `logs/qa_history.log` — question/answer history

---

 
