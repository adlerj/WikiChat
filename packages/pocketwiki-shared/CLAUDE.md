# pocketwiki-shared

Shared schemas, base classes, and utilities used by pocketwiki-builder and pocketwiki-chat.

## Overview

This package provides:
- Pydantic schemas for configuration and state
- Base Stage class for pipeline stages
- Common utilities

## Schemas (`schemas.py`)

### Pipeline Configuration

#### `StreamParseConfig`
Configuration for Wikipedia XML parsing stage.

```python
source_url: FileOrHttpUrl  # http, https, or file:// URLs
output_dir: str
checkpoint_every_pages: int = 1000
checkpoint_every_seconds: int = 60
checkpoint_every_bytes: int = 104857600  # 100 MB
skip_redirects: bool = True
skip_disambiguation: bool = False
force_restart: bool = False
```

#### `ChunkConfig`
Configuration for text chunking.

```python
input_file: str
output_dir: str
max_chunk_tokens: int = 512
overlap_tokens: int = 50
```

#### `FilterConfig`
Configuration for chunk filtering.

```python
input_file: str
output_dir: str
min_chunk_length: int = 100
max_chunk_length: int = 10000
```

#### `EmbedConfig`
Configuration for embedding generation.

```python
input_file: str
output_dir: str
model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
batch_size: int = 32
device: str = "cpu"
```

#### `FAISSConfig`
Configuration for FAISS index creation.

```python
embeddings_file: str
output_dir: str
n_clusters: int = 100
n_subquantizers: int = 96
bits_per_code: int = 8
```

#### `PackageConfig`
Configuration for bundle packaging.

```python
work_dir: str
output_bundle: str
```

### State Tracking

#### `StageState`
Persisted state for pipeline stages.

```python
stage_name: str
input_hash: str
completed: bool
completed_at: Optional[str]
output_files: list[str]
```

#### `StreamParseCheckpoint`
Fine-grained checkpoint for streaming parser.

```python
source_url: FileOrHttpUrl
source_etag: Optional[str]
compressed_bytes_read: int
pages_processed: int
last_page_id: Optional[str]
last_page_title: Optional[str]
output_file: str
output_bytes_written: int
last_checkpoint_time: str
```

## Base Stage Class (`base.py`)

Abstract base class for pipeline stages with:

### Required Methods (Subclasses Implement)

```python
def compute_input_hash(self) -> str:
    """Compute hash of inputs to detect changes."""
    pass

def run(self) -> None:
    """Execute the stage logic."""
    pass
```

### Optional Methods

```python
def get_output_files(self) -> list[Path]:
    """Return list of output files for validation."""
    return []
```

### Provided Methods

```python
def execute(self) -> None:
    """Execute with skip logic and logging."""

def should_skip(self) -> bool:
    """Check if stage can be skipped."""

def persist_state(self) -> None:
    """Save completion state."""

def load_state(self) -> Optional[StageState]:
    """Load previous state."""
```

### Logging Features

The base class provides verbose logging:
- Stage start with config summary
- Input hash computation
- Skip decision with reason
- Completion timing
- Output file sizes

## Usage

```python
from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import ChunkConfig

class ChunkStage(Stage):
    def __init__(self, config: ChunkConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: ChunkConfig = config

    def compute_input_hash(self) -> str:
        # Hash config + input file
        return hashlib.md5(input_content).hexdigest()

    def get_output_files(self) -> list[Path]:
        return [self.output_file]

    def run(self) -> None:
        # Stage implementation
        pass
```

## File Structure

```
packages/pocketwiki-shared/
├── src/pocketwiki_shared/
│   ├── __init__.py
│   ├── base.py      # Stage base class
│   └── schemas.py   # Pydantic schemas
└── pyproject.toml
```

## Testing

```bash
.venv/bin/pytest tests/unit/test_base_stage.py -v
.venv/bin/pytest tests/unit/test_schemas.py -v
```
