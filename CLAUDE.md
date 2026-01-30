# PocketWikiRAG

Offline, portable Wikipedia search + chat system with hybrid retrieval and local LLM.

## Project Structure

```
wikichat/
├── packages/
│   ├── pocketwiki-builder/    # Pipeline to create Wikipedia bundles
│   ├── pocketwiki-chat/       # Chat service with hybrid retrieval
│   └── pocketwiki-shared/     # Shared schemas and base classes
├── crates/
│   └── pocketwiki-python/     # Rust BM25 implementation (via maturin)
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data (sample XML files)
└── Makefile                   # Development commands
```

## Key Technologies

- **FAISS**: Dense vector similarity search (Facebook AI Research)
- **BM25**: Sparse keyword search (Rust implementation via PyO3)
- **SentenceTransformers**: Generate embeddings (all-MiniLM-L6-v2)
- **llama-cpp-python**: Local LLM inference (Llama 3.2 3B)
- **Pydantic**: Schema validation and serialization

## Development Workflow

```bash
# Install all dependencies (creates .venv)
make install

# Run the builder with test fixture
make build ARGS="build --out /tmp/bundle --source-url file://$(pwd)/tests/fixtures/tiny_wiki.xml"

# Run the chat server
make run ARGS="serve --bundle /tmp/bundle/bundle"

# Run all tests
make test

# Clean up
make clean
```

## Architecture Overview

### Builder Pipeline (pocketwiki-builder)

6-stage pipeline that creates portable bundles:

1. **StreamParse**: Stream Wikipedia XML dump, parse articles with checkpointing
2. **Chunk**: Split articles into token-sized chunks
3. **Filter**: Remove low-quality chunks (too short/long)
4. **Embed**: Generate dense embeddings with SentenceTransformer
5. **FAISSIndex**: Build FAISS index (flat for small, IVF-PQ for large)
6. **Package**: Bundle all artifacts with manifest

### Chat Service (pocketwiki-chat)

- Hybrid retrieval: Dense (FAISS) + Sparse (BM25) with RRF fusion
- FastAPI web server with SSE streaming
- Local LLM inference via llama-cpp-python

## Testing

```bash
# Run all tests with coverage
make test

# Run specific test file
.venv/bin/pytest tests/unit/test_cli.py -v

# Run integration tests only
.venv/bin/pytest tests/integration/ -v
```

## Common Tasks

### Adding a new pipeline stage

1. Create stage class in `packages/pocketwiki-builder/src/pocketwiki_builder/pipeline/`
2. Inherit from `Stage` base class
3. Implement `compute_input_hash()` and `run()` methods
4. Add config schema in `packages/pocketwiki-shared/src/pocketwiki_shared/schemas.py`
5. Wire up in `packages/pocketwiki-builder/src/pocketwiki_builder/cli.py`

### Modifying retrieval behavior

- Dense retrieval: `packages/pocketwiki-chat/src/pocketwiki_chat/retrieval/dense.py`
- Sparse retrieval: `packages/pocketwiki-chat/src/pocketwiki_chat/retrieval/sparse.py`
- RRF fusion: `packages/pocketwiki-chat/src/pocketwiki_chat/retrieval/fusion.py`

### Changing LLM prompts

Edit `packages/pocketwiki-chat/src/pocketwiki_chat/llm/prompts.py`
