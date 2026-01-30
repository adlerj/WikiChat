# PocketWikiRAG

Offline, portable Wikipedia search + chat system with hybrid retrieval and local LLM.

## Overview

PocketWikiRAG consists of two applications:

1. **pocketwiki-builder** - Creates a portable bundle from Wikipedia dumps with streaming support
2. **pocketwiki-chat** - Offline chat interface with hybrid retrieval (FAISS + BM25) and local LLM

## Features

- **Streaming indexing** - No 20GB dump download, streams directly from HTTP
- **Checkpoint/resume** - Fine-grained resume at page-level granularity
- **Hybrid retrieval** - Dense (FAISS) + sparse (BM25) with RRF fusion
- **Offline LLM** - Llama 3.2 3B via llama-cpp-python
- **Portable** - 25-30GB bundle fits on USB drive

## Installation

The easiest way to set up the development environment is using the Makefile:

```bash
# Install all dependencies (creates .venv)
make install
```

Or manually:

```bash
# Install all packages in development mode
pip install -e packages/pocketwiki-shared
pip install -e packages/pocketwiki-builder
pip install -e packages/pocketwiki-chat
```

## Usage

### Building a bundle

```bash
pocketwiki-builder build --out my_bundle/
```

The builder will:
- Stream Wikipedia dump from HTTP (no local copy)
- Parse and index incrementally with checkpoints
- Create FAISS dense index + BM25 sparse index
- Compress text blocks with zstd
- Package everything into a portable bundle

### Running the chat interface

```bash
pocketwiki-chat serve --bundle my_bundle/
```

Then open http://localhost:8000 in your browser.

## Development with Makefile

```bash
# Install all dependencies (creates .venv)
make install

# Run the builder (creates Wikipedia bundle)
make build ARGS="build --out ./output"

# Run with local test file
make build ARGS="build --out ./output --source-url file://$(pwd)/tests/fixtures/tiny_wiki.xml"

# Run the chat server
make run ARGS="serve --bundle ./output/bundle"

# Run all tests
make test

# Clean up (removes .venv and caches)
make clean
```

### Running tests

```bash
# Using make
make test

# Or directly with pytest
.venv/bin/pytest tests/ -v

# Run specific test file
.venv/bin/pytest tests/unit/test_cli.py -v

# Run with coverage report
.venv/bin/pytest tests/ --cov
```

### Project structure

See `CLAUDE.md` for detailed architecture documentation.

## License

MIT
