"""Tests for pocketwiki_builder.pipeline.stream_parse."""
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import pytest

from pocketwiki_builder.pipeline.stream_parse import StreamParseStage
from pocketwiki_shared.schemas import StreamParseConfig


class TestStreamParseStage:
    """Tests for StreamParseStage class."""

    def test_init(self, temp_work_dir: Path) -> None:
        """Test stage initialization."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
        )

        stage = StreamParseStage(config, temp_work_dir)

        assert stage.config == config
        assert stage.work_dir == temp_work_dir

    def test_compute_input_hash(self, temp_work_dir: Path) -> None:
        """Test input hash computation."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2"
        )
        stage = StreamParseStage(config, temp_work_dir)

        hash1 = stage.compute_input_hash()
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Same config should produce same hash
        stage2 = StreamParseStage(config, temp_work_dir)
        hash2 = stage2.compute_input_hash()
        assert hash1 == hash2

        # Different config should produce different hash
        config3 = StreamParseConfig(
            source_url="http://example.com/other.xml.bz2"
        )
        stage3 = StreamParseStage(config3, temp_work_dir)
        hash3 = stage3.compute_input_hash()
        assert hash1 != hash3

    @patch("pocketwiki_builder.pipeline.stream_parse.stream_bz2_from_url")
    @patch("pocketwiki_builder.pipeline.stream_parse.WikiXmlParser")
    def test_fresh_parse(
        self,
        mock_parser_class: Mock,
        mock_stream: Mock,
        temp_work_dir: Path,
        sample_articles: list,
    ) -> None:
        """Test fresh parse without checkpoint."""
        # Mock streaming
        mock_stream.return_value = iter([b"<xml>test</xml>"])

        # Mock parser
        mock_parser = Mock()
        mock_parser.parse.return_value = iter(sample_articles)
        mock_parser_class.return_value = mock_parser

        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
        )
        stage = StreamParseStage(config, temp_work_dir)

        stage.run()

        # Should call stream function
        mock_stream.assert_called_once()

        # Should write output file
        output_file = temp_work_dir / "parsed" / "articles.jsonl"
        assert output_file.exists()

        # Verify articles written
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 3

    @patch("pocketwiki_builder.pipeline.stream_parse.stream_bz2_from_url")
    @patch("pocketwiki_builder.pipeline.stream_parse.WikiXmlParser")
    @patch("pocketwiki_builder.pipeline.stream_parse.CheckpointManager")
    def test_parse_with_checkpointing(
        self,
        mock_checkpoint_class: Mock,
        mock_parser_class: Mock,
        mock_stream: Mock,
        temp_work_dir: Path,
        sample_articles: list,
    ) -> None:
        """Test parsing with checkpoint creation."""
        # Mock checkpoint manager
        mock_checkpoint = Mock()
        mock_checkpoint.load_checkpoint.return_value = None
        mock_checkpoint.should_checkpoint.return_value = True
        mock_checkpoint_class.return_value = mock_checkpoint

        # Mock streaming and parsing
        mock_stream.return_value = iter([b"<xml>test</xml>"])
        mock_parser = Mock()
        mock_parser.parse.return_value = iter(sample_articles)
        mock_parser_class.return_value = mock_parser

        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
            checkpoint_every_pages=1,  # Checkpoint after each page
        )
        stage = StreamParseStage(config, temp_work_dir)

        stage.run()

        # Should have called save_checkpoint
        assert mock_checkpoint.save_checkpoint.called

    @patch("pocketwiki_builder.pipeline.stream_parse.stream_bz2_from_url")
    @patch("pocketwiki_builder.pipeline.stream_parse.CheckpointManager")
    def test_resume_from_checkpoint(
        self,
        mock_checkpoint_class: Mock,
        mock_stream: Mock,
        temp_work_dir: Path,
        mock_checkpoint_data: dict,
    ) -> None:
        """Test resuming from checkpoint."""
        from pocketwiki_shared.schemas import StreamParseCheckpoint

        # Mock checkpoint exists
        checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
        mock_checkpoint = Mock()
        mock_checkpoint.load_checkpoint.return_value = checkpoint
        mock_checkpoint.is_checkpoint_valid.return_value = True
        mock_checkpoint_class.return_value = mock_checkpoint

        # Mock streaming with Range request
        mock_stream.return_value = iter([b"<xml>resumed</xml>"])

        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
        )
        stage = StreamParseStage(config, temp_work_dir)

        # Create existing output file
        output_file = temp_work_dir / "parsed" / "articles.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("existing\n")

        stage.run()

        # Should resume with Range request
        mock_stream.assert_called_with(
            "http://example.com/dump.xml.bz2",
            start_byte=checkpoint.compressed_bytes_read,
        )

    def test_force_restart_ignores_checkpoint(
        self, temp_work_dir: Path, mock_checkpoint_data: dict
    ) -> None:
        """Test force_restart option ignores existing checkpoint."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
            force_restart=True,
        )

        with patch(
            "pocketwiki_builder.pipeline.stream_parse.CheckpointManager"
        ) as mock_checkpoint_class:
            mock_checkpoint = Mock()
            from pocketwiki_shared.schemas import StreamParseCheckpoint

            checkpoint = StreamParseCheckpoint(**mock_checkpoint_data)
            mock_checkpoint.load_checkpoint.return_value = checkpoint
            mock_checkpoint_class.return_value = mock_checkpoint

            stage = StreamParseStage(config, temp_work_dir)

            # Should decide not to resume despite checkpoint
            assert stage._should_resume_from_checkpoint() is False

    def test_should_skip_when_completed(self, temp_work_dir: Path) -> None:
        """Test should_skip returns True when already completed."""
        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
        )
        stage = StreamParseStage(config, temp_work_dir)

        # Initially should not skip
        assert stage.should_skip() is False

        # After completion, should skip
        output_file = temp_work_dir / "parsed" / "articles.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("test\n")

        stage.persist_state()

        # New instance should skip
        stage2 = StreamParseStage(config, temp_work_dir)
        assert stage2.should_skip() is True

    @patch("pocketwiki_builder.pipeline.stream_parse.Progress")
    @patch("pocketwiki_builder.pipeline.stream_parse.stream_bz2_from_url")
    @patch("pocketwiki_builder.pipeline.stream_parse.WikiXmlParser")
    def test_progress_display(
        self,
        mock_parser_class: Mock,
        mock_stream: Mock,
        mock_progress: Mock,
        temp_work_dir: Path,
        sample_articles: list,
    ) -> None:
        """Test progress display during parsing."""
        mock_stream.return_value = iter([b"<xml>test</xml>"])
        mock_parser = Mock()
        mock_parser.parse.return_value = iter(sample_articles)
        mock_parser_class.return_value = mock_parser

        config = StreamParseConfig(
            source_url="http://example.com/dump.xml.bz2",
            output_dir=str(temp_work_dir / "parsed"),
        )
        stage = StreamParseStage(config, temp_work_dir)

        stage.run()

        # Progress should be created and used
        mock_progress.assert_called()
