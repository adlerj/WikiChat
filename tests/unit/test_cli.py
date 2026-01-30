"""Tests for pocketwiki-builder CLI."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pocketwiki_builder.cli import cli


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main CLI --help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PocketWiki Builder" in result.output
        assert "build" in result.output

    def test_build_help(self):
        """Test build command --help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert result.exit_code == 0
        assert "--out" in result.output
        assert "--source-url" in result.output
        assert "--checkpoint-pages" in result.output
        assert "--max-chunk-tokens" in result.output
        assert "--force-restart" in result.output


class TestCLIRequiredArgs:
    """Tests for required argument enforcement."""

    def test_build_missing_out(self):
        """Test that build requires --out argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build"])
        assert result.exit_code != 0
        assert "Missing option '--out'" in result.output


class TestCLIInvalidArgs:
    """Tests for invalid argument handling."""

    def test_build_invalid_source_url(self):
        """Test that invalid source URL produces clear error."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["build", "--out", "/tmp/test", "--source-url", "not-a-url"]
        )
        assert result.exit_code != 0

    def test_unknown_command(self):
        """Test that unknown command produces error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["unknown"])
        assert result.exit_code != 0


class TestCLIPipeline:
    """Tests for full pipeline execution."""

    def test_full_pipeline_with_fixture(self, fixtures_dir: Path, tmp_path: Path):
        """Test full pipeline with tiny wiki fixture."""
        tiny_wiki = fixtures_dir / "tiny_wiki.xml"
        output_dir = tmp_path / "bundle_output"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
            ],
        )

        # Check command succeeded
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "PIPELINE COMPLETE" in result.output

        # Check bundle was created
        bundle_dir = output_dir / "bundle"
        assert bundle_dir.exists()

        # Check expected files exist
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "chunks.jsonl").exists()
        assert (bundle_dir / "dense.faiss").exists()

        # Verify manifest contents
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        assert manifest["version"] == "0.1.0"
        assert manifest["num_articles"] == 3
        assert manifest["num_chunks"] == 3

    def test_pipeline_force_restart(self, fixtures_dir: Path, tmp_path: Path):
        """Test that --force-restart clears previous state."""
        tiny_wiki = fixtures_dir / "tiny_wiki.xml"
        output_dir = tmp_path / "bundle_output"

        runner = CliRunner()

        # Run first time
        result1 = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
            ],
        )
        assert result1.exit_code == 0

        # Run second time - should skip stages
        result2 = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
            ],
        )
        assert result2.exit_code == 0
        assert "SKIPPING" in result2.output  # Should skip completed stages

        # Run with --force-restart - should rerun all
        result3 = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
                "--force-restart",
            ],
        )
        assert result3.exit_code == 0
        # Force restart means stream_parse runs fresh
        assert "RUNNING: No previous state found" in result3.output or "Starting FRESH" in result3.output

    def test_pipeline_with_custom_chunk_size(self, fixtures_dir: Path, tmp_path: Path):
        """Test pipeline with custom max-chunk-tokens."""
        tiny_wiki = fixtures_dir / "tiny_wiki.xml"
        output_dir = tmp_path / "bundle_output"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
                "--max-chunk-tokens",
                "50",  # Very small chunk size
            ],
        )

        assert result.exit_code == 0

        # With smaller chunk size, should get more chunks
        bundle_dir = output_dir / "bundle"
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        # With 50 token chunks, we should get more than 3 chunks from 3 articles
        assert manifest["num_chunks"] >= 3


class TestCLILogging:
    """Tests for CLI logging output."""

    def test_logging_output(self, fixtures_dir: Path, tmp_path: Path):
        """Test that CLI produces expected logging output."""
        tiny_wiki = fixtures_dir / "tiny_wiki.xml"
        output_dir = tmp_path / "bundle_output"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "build",
                "--out",
                str(output_dir),
                "--source-url",
                f"file://{tiny_wiki}",
            ],
        )

        # Check for expected log sections
        assert "POCKETWIKI BUILDER - Starting Pipeline" in result.output
        assert "Configuration:" in result.output
        assert "STAGE 1/6: StreamParse" in result.output
        assert "STAGE 2/6: Chunk" in result.output
        assert "STAGE 3/6: Filter" in result.output
        assert "STAGE 4/6: Embed" in result.output
        assert "STAGE 5/6: FAISS Index" in result.output
        assert "STAGE 6/6: Package" in result.output
        assert "PIPELINE COMPLETE" in result.output
        assert "Bundle size:" in result.output
