"""Checkpoint management for streaming parser."""
import hashlib
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pocketwiki_shared.schemas import StreamParseCheckpoint, StreamParseConfig

from .errors import CheckpointError
from .http_stream import get_etag


class CheckpointTrigger(Enum):
    """Checkpoint trigger types."""

    PAGES = "pages"
    BYTES = "bytes"
    TIME = "time"
    MANUAL = "manual"


class CheckpointManager:
    """Manages checkpointing for streaming parser."""

    def __init__(
        self,
        checkpoint_file: Path,
        config: StreamParseConfig,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
            config: Parser configuration
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.config = config
        self.config_hash = self._compute_config_hash()

        # Counters
        self.pages_since_checkpoint = 0
        self.bytes_since_checkpoint = 0
        self.last_checkpoint_time = time.time()

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration.

        Returns:
            Hex string hash
        """
        config_json = self.config.model_dump_json()
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def load_checkpoint(self) -> Optional[StreamParseCheckpoint]:
        """Load checkpoint from file.

        Returns:
            Checkpoint data or None if doesn't exist
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            checkpoint = StreamParseCheckpoint.model_validate_json(
                self.checkpoint_file.read_text()
            )
            return checkpoint
        except Exception:
            # Corrupted checkpoint
            return None

    def save_checkpoint(self, checkpoint: StreamParseCheckpoint) -> None:
        """Save checkpoint atomically.

        Args:
            checkpoint: Checkpoint data to save
        """
        # Add config hash
        checkpoint.config_hash = self.config_hash

        # Write to temp file first
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.checkpoint_file.with_suffix(".json.tmp")

        try:
            temp_file.write_text(checkpoint.model_dump_json(indent=2))
            # Atomic rename
            temp_file.rename(self.checkpoint_file)
        except Exception as e:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def should_checkpoint(
        self, pages_processed: int, bytes_written: int
    ) -> bool:
        """Check if checkpoint should be created.

        Args:
            pages_processed: Total pages processed
            bytes_written: Total bytes written

        Returns:
            True if checkpoint should be created
        """
        # Check page count
        if (
            pages_processed - self.pages_since_checkpoint
            >= self.config.checkpoint_every_pages
        ):
            return True

        # Check bytes written
        if (
            bytes_written - self.bytes_since_checkpoint
            >= self.config.checkpoint_every_bytes
        ):
            return True

        # Check time elapsed
        elapsed = time.time() - self.last_checkpoint_time
        if elapsed >= self.config.checkpoint_every_seconds:
            return True

        return False

    def reset_counters(self) -> None:
        """Reset checkpoint counters after checkpoint created."""
        self.last_checkpoint_time = time.time()
        # Don't reset page/byte counters, use absolute values

    def is_checkpoint_valid(self) -> bool:
        """Check if checkpoint is valid for resuming.

        Returns:
            True if checkpoint can be used for resume
        """
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return False

        # Check config hash
        if checkpoint.config_hash != self.config_hash:
            return False

        # Check source ETag if validation enabled
        if self.config.validate_source_unchanged and checkpoint.source_etag:
            try:
                current_etag = get_etag(str(self.config.source_url))
                if current_etag and current_etag != checkpoint.source_etag:
                    return False
            except Exception:
                # Can't validate, assume invalid
                return False

        return True
