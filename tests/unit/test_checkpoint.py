"""Tests for pocketwiki_builder.streaming.checkpoint."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from pocketwiki_builder.streaming.checkpoint import (
    CheckpointManager,
    CheckpointTrigger,
)
from pocketwiki_shared.schemas import StreamParseCheckpoint, StreamParseConfig


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_init(self, temp_work_dir: Path) -> None:
        """Test checkpoint manager initialization."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(
            checkpoint_file=checkpoint_file,
            config=config,
        )

        assert manager.checkpoint_file == checkpoint_file
        assert manager.pages_since_checkpoint == 0

    def test_load_checkpoint_not_exists(self, temp_work_dir: Path) -> None:
        """Test loading checkpoint when file doesn't exist."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "missing.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)
        checkpoint = manager.load_checkpoint()

        assert checkpoint is None

    def test_save_and_load_checkpoint(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test saving and loading checkpoint."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        # Create checkpoint
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        manager.save_checkpoint(checkpoint)

        # Verify file exists
        assert checkpoint_file.exists()

        # Load and verify
        loaded = manager.load_checkpoint()
        assert loaded is not None
        assert loaded.pages_processed == checkpoint.pages_processed
        assert loaded.last_page_id == checkpoint.last_page_id

    def test_atomic_checkpoint_write(self, temp_work_dir: Path) -> None:
        """Test that checkpoint writes are atomic."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        checkpoint = StreamParseCheckpoint(
            source_url="http://example.com/dump.xml.bz2",
            compressed_bytes_read=1000,
            pages_processed=10,
            output_file="test.jsonl",
            last_checkpoint_time="2026-01-30T10:00:00Z",
        )

        # Save checkpoint
        manager.save_checkpoint(checkpoint)

        # Should not have temp file remaining
        temp_file = checkpoint_file.with_suffix(".json.tmp")
        assert not temp_file.exists()

        # Final file should exist
        assert checkpoint_file.exists()

    def test_should_checkpoint_by_pages(self, temp_work_dir: Path) -> None:
        """Test checkpoint trigger by page count."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            checkpoint_every_pages=100,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        # Not yet
        assert not manager.should_checkpoint(pages_processed=50, bytes_written=0)

        # Trigger at 100
        assert manager.should_checkpoint(pages_processed=100, bytes_written=0)

        # After checkpoint, reset counter
        manager.reset_counters()
        assert not manager.should_checkpoint(pages_processed=150, bytes_written=0)

    def test_should_checkpoint_by_bytes(self, temp_work_dir: Path) -> None:
        """Test checkpoint trigger by bytes written."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            checkpoint_every_bytes=1000,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        # Not yet
        assert not manager.should_checkpoint(pages_processed=10, bytes_written=500)

        # Trigger at 1000
        assert manager.should_checkpoint(pages_processed=10, bytes_written=1000)

    def test_should_checkpoint_by_time(self, temp_work_dir: Path) -> None:
        """Test checkpoint trigger by elapsed time."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            checkpoint_every_seconds=60,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        # Not yet (just started)
        assert not manager.should_checkpoint(pages_processed=10, bytes_written=100)

        # Mock time passage
        with patch("time.time") as mock_time:
            mock_time.return_value = manager.last_checkpoint_time + 61

            # Should trigger after 61 seconds
            assert manager.should_checkpoint(
                pages_processed=10, bytes_written=100
            )

    def test_validate_checkpoint_config_match(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test checkpoint validation with matching config."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        manager.save_checkpoint(checkpoint)

        # Load with same config should succeed
        loaded = manager.load_checkpoint()
        assert loaded is not None

    def test_invalidate_checkpoint_on_config_change(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test checkpoint invalidation when config changes."""
        config1 = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            skip_redirects=True,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager1 = CheckpointManager(checkpoint_file, config1)
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        manager1.save_checkpoint(checkpoint)

        # Load with different config
        config2 = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            skip_redirects=False,  # Changed!
        )
        manager2 = CheckpointManager(checkpoint_file, config2)

        # Should detect config change and invalidate
        assert not manager2.is_checkpoint_valid()

    def test_validate_etag_match(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test ETag validation for source changes."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            validate_source_unchanged=True,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        manager.save_checkpoint(checkpoint)

        # Mock ETag check
        with patch(
            "pocketwiki_builder.streaming.checkpoint.get_etag"
        ) as mock_etag:
            mock_etag.return_value = "abc123"  # Matches checkpoint
            assert manager.is_checkpoint_valid() is True

    def test_invalidate_on_etag_change(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test invalidation when source ETag changes."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            validate_source_unchanged=True,
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        manager.save_checkpoint(checkpoint)

        # Mock ETag change
        with patch(
            "pocketwiki_builder.streaming.checkpoint.get_etag"
        ) as mock_etag:
            mock_etag.return_value = "xyz789"  # Different!
            assert manager.is_checkpoint_valid() is False

    def test_checkpoint_corruption_detection(
        self, temp_work_dir: Path
    ) -> None:
        """Test detection of corrupted checkpoint file."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        # Write corrupted JSON
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text("{invalid json")

        manager = CheckpointManager(checkpoint_file, config)
        checkpoint = manager.load_checkpoint()

        # Should return None for corrupted checkpoint
        assert checkpoint is None

    def test_reset_counters(self, temp_work_dir: Path) -> None:
        """Test resetting checkpoint counters."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        checkpoint_file = temp_work_dir / "checkpoints" / "test.checkpoint.json"

        manager = CheckpointManager(checkpoint_file, config)

        # Simulate some progress
        manager.pages_since_checkpoint = 100
        manager.bytes_since_checkpoint = 5000

        # Reset
        manager.reset_counters()

        assert manager.pages_since_checkpoint == 0
        assert manager.bytes_since_checkpoint == 0


class TestCheckpointTrigger:
    """Tests for CheckpointTrigger enum."""

    def test_trigger_types(self) -> None:
        """Test checkpoint trigger enumeration."""
        assert CheckpointTrigger.PAGES is not None
        assert CheckpointTrigger.BYTES is not None
        assert CheckpointTrigger.TIME is not None
        assert CheckpointTrigger.MANUAL is not None
