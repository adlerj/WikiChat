"""Tests for pocketwiki_shared.schemas."""
import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from pocketwiki_shared.schemas import (
    StreamParseCheckpoint,
    StreamParseConfig,
    StageState,
)


class TestStreamParseCheckpoint:
    """Tests for StreamParseCheckpoint schema."""

    def test_valid_checkpoint(self, mock_checkpoint_data: dict) -> None:
        """Test creating valid checkpoint."""
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        assert checkpoint.pages_processed == 100
        assert checkpoint.last_page_id == "25433"
        assert checkpoint.checkpoint_version == 1

    def test_checkpoint_json_serialization(self, mock_checkpoint_data: dict) -> None:
        """Test checkpoint can be serialized to JSON."""
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        json_str = checkpoint.model_dump_json()
        data = json.loads(json_str)
        assert data["pages_processed"] == 100

    def test_checkpoint_from_json(self, mock_checkpoint_data: dict) -> None:
        """Test checkpoint can be loaded from JSON."""
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        json_str = checkpoint.model_dump_json()
        loaded = StreamParseCheckpoint.model_validate_json(json_str)
        assert loaded.pages_processed == checkpoint.pages_processed
        assert loaded.last_page_id == checkpoint.last_page_id

    def test_checkpoint_validation_missing_fields(self) -> None:
        """Test checkpoint validation fails with missing fields."""
        with pytest.raises(ValidationError):
            StreamParseCheckpoint(source_url="http://example.com")

    def test_checkpoint_validation_invalid_url(self) -> None:
        """Test checkpoint validation fails with invalid URL."""
        data = {
            "source_url": "not a url",
            "compressed_bytes_read": 0,
            "pages_processed": 0,
        }
        with pytest.raises(ValidationError):
            StreamParseCheckpoint(**data)


class TestStreamParseConfig:
    """Tests for StreamParseConfig schema."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = StreamParseConfig(
            source_url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        )
        assert config.checkpoint_every_pages == 1000
        assert config.checkpoint_every_seconds == 60
        assert config.checkpoint_every_bytes == 104857600
        assert config.max_retries == 5
        assert config.skip_redirects is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            checkpoint_every_pages=500,
            max_retries=3,
            force_restart=True,
        )
        assert config.checkpoint_every_pages == 500
        assert config.max_retries == 3
        assert config.force_restart is True

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValidationError):
            StreamParseConfig(
                source_url="http://example.com",
                checkpoint_every_pages=-1,  # Invalid
            )


class TestStageState:
    """Tests for StageState schema."""

    def test_stage_state_creation(self) -> None:
        """Test stage state creation."""
        state = StageState(
            stage_name="stream_parse",
            input_hash="abc123",
            completed=True,
            output_files=["work/parsed/articles.jsonl"],
        )
        assert state.stage_name == "stream_parse"
        assert state.completed is True
        assert len(state.output_files) == 1

    def test_stage_state_serialization(self) -> None:
        """Test stage state JSON serialization."""
        state = StageState(
            stage_name="chunk",
            input_hash="def456",
            completed=False,
            output_files=[],
        )
        json_str = state.model_dump_json()
        loaded = StageState.model_validate_json(json_str)
        assert loaded.stage_name == state.stage_name
        assert loaded.completed is False
