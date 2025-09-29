# Vietnamese PDF RAG — End‑to‑End, Production‑Ready (VN/EN)

Build a recruiter‑ready, end‑to‑end Retrieval‑Augmented Generation (RAG) system for Vietnamese PDF documents: preprocess PDFs → build a hybrid vector DB (dense + sparse) → ask questions via CLI, API, or Streamlit UI with multi‑LLM support (Gemini, Watsonx).

Key strengths:

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
├─ reqs/                     # Requirements per stage
├─ docker-compose.yml        # Optional: Milvus service
├─ .env.example              # Environment template
└─ requirements.txt          # Install all stages at once
```

---

## Quick start

1) Python and environment

- Python 3.10+
- Optional: Docker (for Milvus service)

1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reqs/requirements-preprocess.txt   # stage 1: preprocess
pip install -r reqs/requirements-build.txt        # stage 2: build vector DB
pip install -r reqs/requirements-retrieval.txt    # stage 3: retrieval + UI/API
```

1) Configure environment

Create a `.env` file using `.env.example` as a reference:

```env
# Google Gemini
GEMINI_API_KEY=...

# IBM Watsonx
WATSONX_URL=...
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
```

1) Run the pipeline

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

## Unified architecture (VietnameseRAG)

```python
from src.rag_retriever import VietnameseRAG

rag = VietnameseRAG(
    model_type="gemini",  # or "watsonx"
    k=10,                  # initial retrieval count
    rerank_top_k=5,        # final reranked results
)

# Full RAG
result = rag.answer("Học máy là gì?")
print(result["answer"])

# Only retrieve
docs = rag.search("machine learning")

# Switch model anytime
rag.switch_model("watsonx")
```

Internals (clean modularity):

- DocumentRetriever: dense+sparse hybrid search, RRF fusion, reranking
- AnswerGenerator: LLM prompt generation and answer synthesis

Benefits:

- Single entry point with a clean API
- Hybrid search (BGE‑M3) with RRF + rerank (BGE‑Reranker‑v2‑m3)
- Multi‑LLM via a factory (Gemini, Watsonx) with easy hot‑swap
- Structured logging and Q&A history out of the box

---

## How it works

Hybrid search

- Dense + sparse embeddings via BGE‑M3
- Milvus for vector storage; HNSW (Docker) or IVF_FLAT (local file)
- Reciprocal Rank Fusion (RRF) to combine dense+sparse results
- Optional reranking with BGE‑Reranker‑v2‑m3

LLM integration

- Factory pattern with a simple BaseLLM interface
- Google Gemini (google‑genai) and IBM Watsonx supported
- Consistent `generate(prompt)` API

Vietnamese focus

- Text normalization and cleaning for Vietnamese
- Multilingual embeddings with good VN performance
- Stopwords, UTF‑8 correctness throughout

Logging

- Separate logs per stage (preprocess/build/retrieval/errors)
- QA history log with sources and scores for auditability

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

## What recruiters should notice

- End‑to‑end ownership: ingestion → embeddings → vector DB → hybrid retrieval → LLM answer
- Clear separation of concerns and cohesive APIs
- Robustness: logging, error handling, and environment configuration
- Practical deployment options and documentation that make it easy to run

If you want me to tailor this to your data or infrastructure (GPU/Cloud, new LLMs, or custom evaluators), it’s designed to extend cleanly.
