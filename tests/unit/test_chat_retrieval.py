"""Tests for pocketwiki_chat retrieval components."""
from pathlib import Path
from unittest.mock import Mock, patch
import json

import pytest
import numpy as np


class TestBundleLoader:
    """Tests for bundle loading."""

    def test_load_bundle(self, temp_work_dir: Path) -> None:
        """Test loading a bundle."""
        from pocketwiki_chat.bundle.loader import BundleLoader

        # Create fake bundle
        bundle_dir = temp_work_dir / "bundle"
        bundle_dir.mkdir()

        manifest = {
            "version": "0.1.0",
            "created_at": "2026-01-30T10:00:00Z",
            "num_articles": 100,
            "num_chunks": 500,
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        loader = BundleLoader(bundle_dir)
        loaded_manifest = loader.load_manifest()

        assert loaded_manifest["num_articles"] == 100

    def test_validate_bundle(self, temp_work_dir: Path) -> None:
        """Test bundle validation."""
        from pocketwiki_chat.bundle.loader import BundleLoader

        bundle_dir = temp_work_dir / "bundle"
        bundle_dir.mkdir()

        # Missing manifest should fail validation
        loader = BundleLoader(bundle_dir)
        assert loader.validate() is False


class TestDenseRetrieval:
    """Tests for FAISS dense retrieval."""

    @patch("faiss.read_index")
    @patch("sentence_transformers.SentenceTransformer")
    def test_dense_search(
        self, mock_model_class: Mock, mock_faiss: Mock, temp_work_dir: Path
    ) -> None:
        """Test dense vector search."""
        from pocketwiki_chat.retrieval.dense import DenseRetriever

        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # distances
            np.array([[0, 1, 2]]),  # indices
        )
        mock_faiss.return_value = mock_index

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_model_class.return_value = mock_model

        index_path = temp_work_dir / "dense.faiss"
        index_path.write_text("fake")

        retriever = DenseRetriever(index_path, "all-MiniLM-L6-v2")
        results = retriever.search("test query", k=3)

        assert len(results) == 3
        assert results[0]["rank"] == 0


class TestSparseRetrieval:
    """Tests for BM25 sparse retrieval."""

    def test_bm25_search(self, temp_work_dir: Path) -> None:
        """Test BM25 search."""
        from pocketwiki_chat.retrieval.sparse import SparseRetriever

        # Would test BM25 implementation
        # Requires Rust components to be built
        pass


class TestRRFFusion:
    """Tests for RRF fusion."""

    def test_rrf_fusion(self) -> None:
        """Test reciprocal rank fusion."""
        from pocketwiki_chat.retrieval.fusion import rrf_fusion

        dense_results = [
            {"chunk_id": "1", "score": 0.9, "rank": 0},
            {"chunk_id": "2", "score": 0.8, "rank": 1},
            {"chunk_id": "3", "score": 0.7, "rank": 2},
        ]

        sparse_results = [
            {"chunk_id": "2", "score": 10.0, "rank": 0},
            {"chunk_id": "1", "score": 8.0, "rank": 1},
            {"chunk_id": "4", "score": 6.0, "rank": 2},
        ]

        fused = rrf_fusion(dense_results, sparse_results, k=60)

        assert len(fused) == 4  # Unique chunks
        # Chunk 2 appears high in both, should rank first
        assert fused[0]["chunk_id"] == "2"

    def test_rrf_formula(self) -> None:
        """Test RRF score calculation."""
        from pocketwiki_chat.retrieval.fusion import rrf_score

        # score = 1/(k + rank)
        score = rrf_score(rank=0, k=60)
        assert score == 1.0 / 60

        score = rrf_score(rank=10, k=60)
        assert score == 1.0 / 70


class TestContextAssembly:
    """Tests for context assembly."""

    def test_assemble_context(self) -> None:
        """Test assembling context from chunks."""
        from pocketwiki_chat.retrieval.context import assemble_context

        chunks = [
            {
                "chunk_id": "1",
                "text": "Albert Einstein was a physicist.",
                "page_title": "Albert Einstein",
                "page_id": "736",
            },
            {
                "chunk_id": "2",
                "text": "Python is a programming language.",
                "page_title": "Python",
                "page_id": "23862",
            },
        ]

        context = assemble_context(chunks, max_tokens=500)

        assert "Albert Einstein" in context
        assert "Python" in context

    def test_context_truncation(self) -> None:
        """Test context is truncated to max tokens."""
        from pocketwiki_chat.retrieval.context import assemble_context

        chunks = [{"chunk_id": str(i), "text": "A" * 1000} for i in range(10)]

        context = assemble_context(chunks, max_tokens=100)

        # Should be truncated
        assert len(context) < 10 * 1000
