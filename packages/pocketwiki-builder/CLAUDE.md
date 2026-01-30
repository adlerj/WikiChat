# pocketwiki-builder

Pipeline for creating portable Wikipedia bundles from MediaWiki XML dumps.

## Overview

The builder processes Wikipedia dumps through a 6-stage pipeline with checkpointing support. Each stage persists state to enable resume after interruption.

## Pipeline Stages

```
StreamParse → Chunk → Filter → Embed → FAISSIndex → Package
```

### 1. StreamParse (`pipeline/stream_parse.py`)

Streams and parses Wikipedia XML dumps with checkpoint support.

- **Input**: Wikipedia XML dump URL (http/https/file)
- **Output**: `work/parsed/articles.jsonl`
- **Features**:
  - HTTP range requests for resume
  - bz2 decompression on-the-fly
  - Checkpoint every N pages/bytes/seconds
  - Skip redirects and disambiguation pages
  - Namespace filtering

### 2. Chunk (`pipeline/chunk.py`)

Splits articles into token-sized chunks for embedding.

- **Input**: `work/parsed/articles.jsonl`
- **Output**: `work/chunks/chunks.jsonl`
- **Config**: `max_chunk_tokens` (default: 512)

### 3. Filter (`pipeline/filter.py`)

Removes low-quality chunks.

- **Input**: `work/chunks/chunks.jsonl`
- **Output**: `work/filtered/filtered.jsonl`
- **Criteria**:
  - `min_chunk_length`: 100 chars
  - `max_chunk_length`: 10000 chars

### 4. Embed (`pipeline/embed.py`)

Generates dense embeddings using SentenceTransformer.

- **Input**: `work/filtered/filtered.jsonl`
- **Output**: `work/embeddings/embeddings.npy`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)

### 5. FAISSIndex (`pipeline/faiss_index.py`)

Creates FAISS index for similarity search.

- **Input**: `work/embeddings/embeddings.npy`
- **Output**: `work/indexes/dense.faiss`
- **Index types**:
  - Small datasets (<200 vectors): IndexFlatIP (exact)
  - Large datasets: IVF-PQ (approximate, trained)

### 6. Package (`pipeline/package.py`)

Bundles all artifacts for distribution.

- **Input**: work directory
- **Output**: `bundle/` directory with manifest

## Data Flow

```
Wikipedia XML → articles.jsonl → chunks.jsonl → filtered.jsonl
                                                      ↓
                                               embeddings.npy
                                                      ↓
                                               dense.faiss → bundle/
```

## Checkpoint/Resume Behavior

Each stage tracks:
- **Input hash**: Detects if inputs changed
- **Completion state**: Whether stage finished
- **Output files**: Validates outputs exist

On re-run:
1. Compute current input hash
2. Compare with stored state
3. Skip if hash matches and outputs exist
4. Re-run if anything changed

StreamParse has additional checkpoint granularity:
- Saves progress every N pages
- Can resume from last checkpoint on interruption

## Configuration

All config via CLI flags or Pydantic schemas in `pocketwiki_shared.schemas`.

Key configs:
- `--source-url`: Wikipedia dump URL
- `--out`: Output directory
- `--checkpoint-pages`: Pages between checkpoints
- `--max-chunk-tokens`: Chunk size
- `--force-restart`: Ignore previous state

## CLI Usage

```bash
# Full Wikipedia build (streams from Wikimedia)
pocketwiki-builder build --out ./output

# Local file (for testing)
pocketwiki-builder build --out ./output --source-url file:///path/to/wiki.xml

# Force fresh start
pocketwiki-builder build --out ./output --force-restart
```

## File Structure

```
packages/pocketwiki-builder/
├── src/pocketwiki_builder/
│   ├── cli.py              # Click CLI entry point
│   ├── pipeline/
│   │   ├── stream_parse.py # Stage 1
│   │   ├── chunk.py        # Stage 2
│   │   ├── filter.py       # Stage 3
│   │   ├── embed.py        # Stage 4
│   │   ├── faiss_index.py  # Stage 5
│   │   └── package.py      # Stage 6
│   └── streaming/
│       ├── checkpoint.py   # Checkpoint management
│       ├── http_stream.py  # HTTP streaming with bz2
│       └── xml_parser.py   # MediaWiki XML parser
└── pyproject.toml
```
