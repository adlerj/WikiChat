"""Tests for remaining pipeline stages."""
from pathlib import Path
from unittest.mock import Mock, patch
import json

import pytest
import numpy as np


class TestChunkStage:
    """Tests for chunking stage."""

    def test_chunk_articles(self, temp_work_dir: Path) -> None:
        """Test chunking articles into smaller pieces."""
        from pocketwiki_builder.pipeline.chunk import ChunkStage, ChunkConfig

        # Create input file
        input_file = temp_work_dir / "parsed" / "articles.jsonl"
        input_file.parent.mkdir(parents=True, exist_ok=True)

        articles = [
            {"id": "1", "title": "Test", "text": "A" * 1000},
            {"id": "2", "title": "Test2", "text": "B" * 2000},
        ]
        input_file.write_text("\n".join(json.dumps(a) for a in articles))

        config = ChunkConfig(
            input_file=str(input_file),
            output_dir=str(temp_work_dir / "chunks"),
            max_chunk_tokens=200,
        )

        stage = ChunkStage(config, temp_work_dir)
        stage.run()

        # Should create chunks file
        output_file = temp_work_dir / "chunks" / "chunks.jsonl"
        assert output_file.exists()

        # Verify chunks
        chunks = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(chunks) > 2  # Should be split into multiple chunks


class TestFilterStage:
    """Tests for filtering stage."""

    def test_filter_low_quality(self, temp_work_dir: Path) -> None:
        """Test filtering low-quality chunks."""
        from pocketwiki_builder.pipeline.filter import FilterStage, FilterConfig

        # Create input file with mixed quality
        input_file = temp_work_dir / "chunks" / "chunks.jsonl"
        input_file.parent.mkdir(parents=True, exist_ok=True)

        chunks = [
            {"id": "1", "text": "High quality content with substance", "page_title": "Good"},
            {"id": "2", "text": "stub", "page_title": "Bad"},  # Too short
            {"id": "3", "text": "Another good piece of content here", "page_title": "Good2"},
        ]
        input_file.write_text("\n".join(json.dumps(c) for c in chunks))

        config = FilterConfig(
            input_file=str(input_file),
            output_dir=str(temp_work_dir / "filtered"),
            min_chunk_length=20,
        )

        stage = FilterStage(config, temp_work_dir)
        stage.run()

        # Should filter out short chunks
        output_file = temp_work_dir / "filtered" / "filtered.jsonl"
        assert output_file.exists()

        filtered = [json.loads(line) for line in output_file.read_text().strip().split("\n")]
        assert len(filtered) == 2  # Only 2 good chunks


class TestEmbedStage:
    """Tests for embedding stage."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_chunks(self, mock_model_class: Mock, temp_work_dir: Path) -> None:
        """Test embedding chunks with sentence-transformers."""
        from pocketwiki_builder.pipeline.embed import EmbedStage, EmbedConfig

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        mock_model_class.return_value = mock_model

        # Create input
        input_file = temp_work_dir / "filtered" / "filtered.jsonl"
        input_file.parent.mkdir(parents=True, exist_ok=True)

        chunks = [
            {"id": "1", "text": "Test chunk one"},
            {"id": "2", "text": "Test chunk two"},
        ]
        input_file.write_text("\n".join(json.dumps(c) for c in chunks))

        config = EmbedConfig(
            input_file=str(input_file),
            output_dir=str(temp_work_dir / "embeddings"),
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
        )

        stage = EmbedStage(config, temp_work_dir)
        stage.run()

        # Should create embeddings file
        output_file = temp_work_dir / "embeddings" / "embeddings.npy"
        assert output_file.exists()


class TestFAISSIndexStage:
    """Tests for FAISS indexing."""

    @patch("faiss.IndexIVFPQ")
    @patch("faiss.IndexFlatL2")
    def test_create_faiss_index(
        self, mock_flat: Mock, mock_ivfpq: Mock, temp_work_dir: Path
    ) -> None:
        """Test FAISS IVF-PQ index creation."""
        from pocketwiki_builder.pipeline.faiss_index import FAISSIndexStage, FAISSConfig

        # Create embeddings
        embeddings_file = temp_work_dir / "embeddings" / "embeddings.npy"
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.random.rand(1000, 384).astype(np.float32)
        np.save(embeddings_file, embeddings)

        config = FAISSConfig(
            embeddings_file=str(embeddings_file),
            output_dir=str(temp_work_dir / "indexes"),
            n_clusters=100,
        )

        stage = FAISSIndexStage(config, temp_work_dir)
        # Would test actual index creation if not mocked


class TestPackageStage:
    """Tests for packaging stage."""

    def test_create_bundle(self, temp_work_dir: Path) -> None:
        """Test creating final bundle with manifest."""
        from pocketwiki_builder.pipeline.package import PackageStage, PackageConfig

        # Create dummy files
        (temp_work_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (temp_work_dir / "indexes" / "dense.faiss").write_text("fake")
        (temp_work_dir / "indexes" / "sparse.dict").write_text("fake")

        config = PackageConfig(
            work_dir=str(temp_work_dir),
            output_bundle=str(temp_work_dir / "bundle"),
        )

        stage = PackageStage(config, temp_work_dir)
        stage.run()

        # Should create manifest
        manifest = temp_work_dir / "bundle" / "manifest.json"
        assert manifest.exists()

        data = json.loads(manifest.read_text())
        assert "version" in data
