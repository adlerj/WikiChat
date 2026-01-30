"""Shared Pydantic schemas for PocketWiki."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StreamParseCheckpoint(BaseModel):
    """Checkpoint data for streaming parser."""

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )

    source_url: HttpUrl
    source_etag: Optional[str] = None
    compressed_bytes_read: int = Field(ge=0, default=0)
    pages_processed: int = Field(ge=0, default=0)
    last_page_id: Optional[str] = None
    last_page_title: Optional[str] = None
    output_file: str = ""
    output_bytes_written: int = Field(ge=0, default=0)
    last_checkpoint_time: str = ""  # ISO format datetime
    checkpoint_version: int = 1
    config_hash: Optional[str] = None


class StreamParseConfig(BaseModel):
    """Configuration for StreamParse stage."""

    source_url: HttpUrl
    output_dir: str = "work/parsed"
    output_filename: str = "articles.jsonl"

    # Checkpointing
    checkpoint_every_pages: int = Field(default=1000, ge=1)
    checkpoint_every_seconds: int = Field(default=60, ge=1)
    checkpoint_every_bytes: int = Field(default=104857600, ge=1)  # 100 MB

    # HTTP streaming
    http_chunk_size: int = Field(default=1048576, ge=1024)  # 1 MB
    http_timeout: int = Field(default=300, ge=1)  # 5 minutes

    # Retry behavior
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: int = Field(default=10, ge=1)

    # Parsing
    skip_redirects: bool = True
    skip_disambiguation: bool = False
    allowed_namespaces: list[int] = Field(default_factory=lambda: [0])

    # Resume behavior
    force_restart: bool = False
    validate_source_unchanged: bool = True


class StageState(BaseModel):
    """State persisted by a pipeline stage."""

    stage_name: str
    input_hash: str
    completed: bool = False
    completed_at: Optional[str] = None
    output_files: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class StageConfig(BaseModel):
    """Base configuration for stages."""

    pass


class ChunkConfig(StageConfig):
    """Configuration for chunking stage."""

    input_file: str
    output_dir: str
    max_chunk_tokens: int = Field(default=512, ge=1)
    overlap_tokens: int = Field(default=50, ge=0)


class FilterConfig(StageConfig):
    """Configuration for filtering stage."""

    input_file: str
    output_dir: str
    min_chunk_length: int = Field(default=100, ge=0)
    max_chunk_length: int = Field(default=10000, ge=1)


class EmbedConfig(StageConfig):
    """Configuration for embedding stage."""

    input_file: str
    output_dir: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1)
    device: str = "cpu"


class FAISSConfig(StageConfig):
    """Configuration for FAISS indexing."""

    embeddings_file: str
    output_dir: str
    n_clusters: int = Field(default=100, ge=1)
    n_subquantizers: int = Field(default=96, ge=1)
    bits_per_code: int = Field(default=8, ge=1)


class PackageConfig(StageConfig):
    """Configuration for packaging stage."""

    work_dir: str
    output_bundle: str
