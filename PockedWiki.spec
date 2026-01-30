# PocketWikiRAG Implementation Plan

## Overview

Build an offline, portable Wikipedia search + chat system with two apps:

1. **pocketwiki_builder** - Creates a portable bundle from Wikipedia dump
2. **pocketwiki_chat** - Offline chat interface with hybrid retrieval + local LLM

## Technical Stack (Confirmed)

- **Language:** Python + Rust hybrid (PyO3 + maturin)
- **Embedding:** all-MiniLM-L6-v2 (384-dim)
- **LLM:** Llama 3.2 3B GGUF via llama-cpp-python
- **Wiki parsing:** wikiextractor
- **Distribution:** pip installable
- **UI:** Web UI only (FastAPI + vanilla JS)
- **Platform:** Cross-platform

---

## Project Structure

```
pocketwiki/
├── pyproject.toml                    # Workspace root
├── crates/                           # Rust workspace
│   ├── Cargo.toml
│   ├── pocketwiki-core/              # Core Rust library
│   │   └── src/
│   │       ├── bm25/                 # BM25 inverted index
│   │       ├── text_store/           # Block compression/decompression
│   │       └── tokenizer/            # Simple tokenizer
│   └── pocketwiki-python/            # PyO3 bindings
├── packages/
│   ├── pocketwiki-shared/            # Shared schemas, utils
│   ├── pocketwiki-builder/           # Builder CLI + pipeline
│   │   └── src/pocketwiki_builder/
│   │       ├── cli.py
│   │       ├── pipeline/             # Resumable stages
│   │       │   ├── stream_parse.py   # StreamParse (merged download+parse)
│   │       │   ├── chunk.py
│   │       │   ├── filter.py
│   │       │   ├── embed.py
│   │       │   ├── faiss_index.py
│   │       │   ├── sparse_index.py
│   │       │   ├── text_store.py
│   │       │   └── package.py
│   │       ├── streaming/            # Streaming utilities
│   │       │   ├── http_stream.py    # HTTP Range + bz2 decompression
│   │       │   ├── xml_parser.py     # Incremental lxml parsing
│   │       │   ├── checkpoint.py     # CheckpointManager
│   │       │   └── errors.py         # Retry logic
│   │       └── wiki/                 # wikiextractor wrapper
│   └── pocketwiki-chat/              # Chat app
│       └── src/pocketwiki_chat/
│           ├── cli.py
│           ├── bundle/               # Bundle loader
│           ├── retrieval/            # Dense, sparse, fusion
│           ├── llm/                  # llama-cpp integration
│           └── web/                  # FastAPI + static files
├── tests/
│   ├── fixtures/                     # Sample dump + bundle
│   ├── unit/
│   └── integration/
└── scripts/
```

---

## Rust Components (Performance-Critical)

| Component                | Rationale                                                    |
| ------------------------ | ------------------------------------------------------------ |
| **BM25 Inverted Index**  | Hot path; compressed postings need efficient varint decoding |
| **Postings Compression** | Delta encoding + varint for space efficiency                 |
| **Text Block Reader**    | zstd decompression with LRU cache                            |
| **Simple Tokenizer**     | Must match index-time exactly                                |

**Keep in Python:** FAISS, embedding model, wiki parsing, pipeline orchestration, web UI, LLM integration

---

## Key Python Dependencies

| Library                 | Purpose                  |
| ----------------------- | ------------------------ |
| `faiss-cpu`             | Dense ANN index (IVF-PQ) |
| `sentence-transformers` | Embedding model          |
| `llama-cpp-python`      | Local LLM inference      |
| `wikiextractor`         | Wikipedia XML parsing    |
| `fastapi` + `uvicorn`   | Web API + server         |
| `click` + `rich`        | CLI + progress bars      |
| `pydantic`              | Data validation          |
| `pyarrow`               | Parquet support          |

---

## Builder Pipeline (Resumable Stages)

```
StreamParse → Chunk → Filter → Embed → FAISS Index
                                     ↘
                                Sparse Index → Text Store → Package
```

Each stage:

- Persists state to `{stage}.state.json`
- Computes input hash to detect changes
- Skips if already completed with same inputs

### Stage 1: StreamParse (Merged Download + Parse)

**Streams Wikipedia dump directly from HTTP, decompresses bz2 on-the-fly, parses XML incrementally, and writes articles to JSONL with fine-grained checkpointing.**

**Key features:**
- HTTP Range requests for resume support
- bz2 decompression in-stream (no disk copy of dump)
- Incremental XML parsing with lxml.etree.iterparse()
- Checkpoint every N pages (default: 1000) for fine-grained resume
- ETag validation to detect source changes

**Input:** HTTP URL to bz2-compressed Wikipedia dump
**Output:** `work/parsed/articles.jsonl` (one article per line)
**Checkpoint:** `work/checkpoints/stream_parse.checkpoint.json`

**Checkpoint schema:**
```json
{
  "source_url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
  "source_etag": "abc123...",
  "compressed_bytes_read": 1073741824,
  "pages_processed": 50000,
  "last_page_id": "123456",
  "last_page_title": "Albert Einstein",
  "output_file": "work/parsed/articles.jsonl",
  "output_bytes_written": 2147483648,
  "last_checkpoint_time": "2026-01-30T10:30:00Z",
  "checkpoint_version": 1
}
```

**Configuration:**
```toml
[tool.pocketwiki.builder.stream_parse]
source_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
output_dir = "work/parsed"
checkpoint_every_pages = 1000
checkpoint_every_seconds = 60
checkpoint_every_bytes = 104857600  # 100 MB
http_chunk_size = 1048576  # 1 MB
http_timeout = 300  # 5 minutes
max_retries = 5
retry_backoff_seconds = 10
skip_redirects = true
validate_source_unchanged = true
force_restart = false
```

**Resume behavior:**
- Validates checkpoint matches current config
- Checks source ETag to detect dump changes
- Uses HTTP Range to resume from byte offset
- Skips already-processed pages using last_page_id
- Atomic checkpoint writes prevent corruption

**Benefits:**
- **20 GB disk savings** - no local dump copy needed
- **30-60 min time savings** - overlapped download + parse
- **Fine-grained resume** - < 1 min lost work vs full stage restart
- **Robust error handling** - exponential backoff, retry up to 5x

**Bundle Output:**

- `manifest.json` - versions, params, sizes
- `dense.faiss` + `dense.meta` - FAISS IVF-PQ index
- `sparse.dict` + `sparse.postings` + `sparse.meta` - BM25 index
- `text.zstblocks` + `text.index` - compressed text
- `meta.sqlite` - pages + chunks metadata
- `models/` - embedding model files

---

## Streaming Implementation Details

### HTTP Streaming with Decompression

```python
import requests
import bz2

def stream_bz2_from_url(url: str, start_byte: int = 0) -> Iterator[bytes]:
    """Stream bz2-compressed data from URL with resume support."""
    headers = {'Range': f'bytes={start_byte}-'} if start_byte > 0 else {}

    with requests.get(url, stream=True, headers=headers) as response:
        response.raise_for_status()
        decompressor = bz2.BZ2Decompressor()

        for chunk in response.iter_content(chunk_size=1024*1024):  # 1 MB
            if chunk:
                yield decompressor.decompress(chunk)
```

### Incremental XML Parsing

```python
from lxml import etree

def parse_wiki_xml_stream(byte_stream: Iterator[bytes]) -> Iterator[Dict]:
    """Parse Wikipedia XML incrementally to prevent memory leaks."""
    context = etree.iterparse(
        byte_stream,
        events=('end',),
        tag='{http://www.mediawiki.org/xml/export-0.10/}page'
    )

    for event, elem in context:
        page_id = elem.findtext('.//{*}id')
        title = elem.findtext('.//{*}title')
        text = elem.findtext('.//{*}revision/{*}text')

        yield {'id': page_id, 'title': title, 'text': text}

        # Critical: Clear element to prevent memory leak
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
```

### Error Handling

- **Network errors:** Exponential backoff retry (max 5 attempts)
- **HTTP 5xx:** Retry with backoff
- **HTTP 4xx:** Fail immediately (client error)
- **Malformed XML:** Log warning, skip page, continue
- **Any error:** Write checkpoint before failing

### Resume Decision Flow

1. Check if checkpoint exists → No? Start fresh
2. Validate config hash matches → Changed? Start fresh
3. Check source ETag (optional) → Changed? Start fresh
4. Verify output file size matches → Mismatch? Start fresh
5. All checks pass → Resume from checkpoint using HTTP Range

---

## Chat Retrieval Flow

```
Query → Embed (dense) + Tokenize (sparse)
          ↓                    ↓
     FAISS top-200        BM25 top-200
          ↓                    ↓
          └──── RRF Fusion ────┘
                    ↓
              Top-20 chunks
                    ↓
         Decompress text blocks
                    ↓
           Context assembly
                    ↓
         LLM generation (streaming)
                    ↓
         Response with citations
```

**RRF Formula:** `score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)` where k=60

---

## Web UI Architecture

- **FastAPI** serves API routes + static files
- **Vanilla JS** (no React) for simplicity
- **SSE** for streaming LLM responses
- **Two-panel layout:** Chat + Sources

**Routes:**

- `GET /` → index.html
- `POST /api/chat` → SSE stream (sources + tokens)
- `POST /api/search` → search-only results
- `GET /api/page/{id}` → full page text

---

## Size Estimates (Full Wikipedia)

**Final bundle size:**

| Component            | Size          |
| -------------------- | ------------- |
| Dense index (IVF-PQ) | ~5 GB         |
| Sparse index (BM25)  | ~3-5 GB       |
| Text blocks (zstd)   | ~15-20 GB     |
| SQLite metadata      | ~1-2 GB       |
| Embedding model      | ~100 MB       |
| **Total**            | **~25-30 GB** |

**Peak disk usage during build:**

- **With streaming (new):** ~90 GB (parsed articles.jsonl only, no dump copy)
- **Old approach:** ~110 GB (20 GB dump + 90 GB parsed)
- **Savings:** 20 GB (no local dump copy needed)

Fits on 64GB USB with room for LLM model (~2-4 GB)

---

## Implementation Phases

### Phase 1: Foundation

- Set up monorepo with pyproject.toml files
- Create `pocketwiki-shared` schemas
- Set up Rust workspace skeleton
- Configure CI for all platforms

### Phase 2: Builder Core

- StreamParse stage with HTTP streaming + checkpoint/resume
  - HTTP Range requests with retry logic
  - bz2 decompression on-the-fly
  - Incremental XML parsing with lxml
  - Fine-grained checkpointing (every N pages/seconds/bytes)
- wikiextractor integration
- Chunking with token counting
- Size filtering

### Phase 3: Indexing

- Embedding with batching
- FAISS IVF-PQ index
- Rust BM25 + PyO3 bindings
- Text block compression

### Phase 4: Chat Core

- Bundle loader/validator
- Dense + sparse retrieval
- RRF fusion
- Context assembly

### Phase 5: LLM and UI

- llama-cpp-python integration
- RAG prompt templates
- FastAPI application
- Web UI (HTML/JS/CSS)

### Phase 6: Polish

- End-to-end testing
- Performance optimization
- Documentation

---

## Testing Strategy

- **Fixtures:** ~100 article sample dump + pre-built bundle
- **Unit tests:**
  - BM25, chunking, fusion, manifest
  - HTTP streaming with bz2 decompression
  - Incremental XML parsing with memory leak prevention
  - Checkpoint atomic writes and resume logic
- **Integration tests:**
  - Full pipeline, hybrid retrieval, end-to-end
  - StreamParse resume after simulated crash
  - Config change invalidates checkpoint
  - Source ETag change triggers restart
- **CI:** Test on Ubuntu, macOS, Windows with Python 3.10-3.12

---

## Immediate Tasks (Before Implementation)

1. **Update spec file** - Write confirmed technical decisions back to `PockedWiki.spec`
2. **Initialize git repo** - `git init` and commit the updated spec

---

## Verification Plan

1. **StreamParse verification:**
   - Verify HTTP streaming + bz2 decompression works
   - Check checkpoint file created every N pages
   - Simulate crash (SIGKILL) and verify resume from checkpoint
   - Verify no 20GB dump file exists on disk
   - Check memory usage stays < 500 MB during parse
   - Verify articles.jsonl output matches expected format

2. **Builder verification:**
   - Run on sample dump, verify bundle structure
   - Query sparse index for known article title
   - Query dense index for semantic similarity
   - Verify peak disk usage ~90 GB (not 110 GB)

3. **Chat verification:**
   - Load bundle, verify all components
   - Test hybrid search returns relevant chunks
   - Test LLM generates responses with citations
   - Test search-only mode

4. **End-to-end:**
   - Build sample bundle → Load in chat → Query → Verify citations
