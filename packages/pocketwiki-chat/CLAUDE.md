# pocketwiki-chat

Chat service with hybrid retrieval (FAISS + BM25) and local LLM inference.

## Overview

Serves a web interface for querying Wikipedia bundles created by pocketwiki-builder. Uses hybrid retrieval combining dense semantic search with sparse keyword search.

## Architecture

```
User Query → Embedding → [Dense Search] ─┐
                                         ├─→ RRF Fusion → Context → LLM → Response
User Query → Tokenize  → [BM25 Search]  ─┘
```

## Components

### Bundle Loader (`bundle/loader.py`)

Loads and validates bundle contents:
- `manifest.json`: Bundle metadata
- `chunks.jsonl`: Text chunks
- `dense.faiss`: FAISS index
- `sparse.dict` / `sparse.postings`: BM25 index (optional)

### Retrieval System (`retrieval/`)

#### Dense Retrieval (`dense.py`)

Uses FAISS for semantic similarity search.

- Loads SentenceTransformer model
- Embeds query
- Searches FAISS index for nearest neighbors
- Returns top-k chunk IDs with scores

#### Sparse Retrieval (`sparse.py`)

Uses Rust BM25 implementation for keyword search.

- Loads BM25 index from bundle
- Tokenizes query
- Returns top-k chunk IDs with BM25 scores

#### RRF Fusion (`fusion.py`)

Combines dense and sparse results using Reciprocal Rank Fusion.

```python
RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
```

Default k=60 provides good balance.

#### Context Assembly (`context.py`)

Assembles retrieved chunks into LLM context:
- Deduplicates chunks
- Orders by relevance
- Truncates to fit context window

### LLM Integration (`llm/`)

#### Generator (`generator.py`)

Manages llama-cpp-python model:
- Lazy loading on first query
- Streaming token generation
- Configurable parameters (temperature, max_tokens)

#### Prompts (`prompts.py`)

System prompts for RAG responses.

### Web API (`web/app.py`)

FastAPI application with:

**Endpoints:**
- `GET /`: Serve web UI
- `POST /api/search`: Search without LLM
- `POST /api/chat`: Full RAG with LLM
- `GET /api/chat/stream`: SSE streaming responses
- `GET /health`: Health check

**SSE Streaming:**
Streams tokens as Server-Sent Events for real-time UI updates.

## CLI Usage

```bash
# Start server
pocketwiki-chat serve --bundle ./path/to/bundle

# Custom port
pocketwiki-chat serve --bundle ./path/to/bundle --port 8080

# With specific LLM model
pocketwiki-chat serve --bundle ./path/to/bundle --model ./models/llama.gguf
```

## Configuration

Environment variables:
- `POCKETWIKI_BUNDLE`: Default bundle path
- `POCKETWIKI_MODEL`: Default LLM model path

CLI options:
- `--bundle`: Bundle directory (required)
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: 0.0.0.0)
- `--model`: LLM model file path

## File Structure

```
packages/pocketwiki-chat/
├── src/pocketwiki_chat/
│   ├── cli.py              # Click CLI entry point
│   ├── bundle/
│   │   └── loader.py       # Bundle loading/validation
│   ├── retrieval/
│   │   ├── dense.py        # FAISS retrieval
│   │   ├── sparse.py       # BM25 retrieval
│   │   ├── fusion.py       # RRF fusion
│   │   └── context.py      # Context assembly
│   ├── llm/
│   │   ├── generator.py    # LLM wrapper
│   │   └── prompts.py      # System prompts
│   └── web/
│       ├── app.py          # FastAPI application
│       └── templates/      # HTML templates
└── pyproject.toml
```

## Retrieval Tuning

Adjust in code:
- `k` parameter in RRF (fusion.py) - higher = more weight to lower ranks
- `top_k` for dense/sparse search - more candidates = better fusion
- Context token limit - balance between context and response length

## Testing

```bash
# Test retrieval
.venv/bin/pytest tests/unit/test_chat_retrieval.py -v

# Test web API
.venv/bin/pytest tests/unit/test_web_api.py -v
```
