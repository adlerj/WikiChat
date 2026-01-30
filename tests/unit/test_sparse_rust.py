"""Tests for Rust BM25 integration."""
import json
import pytest
from pathlib import Path

try:
    from pocketwiki_rust import BM25Index
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestRustBM25:
    """Test Rust BM25 implementation."""

    def test_basic_index_and_search(self):
        """Test creating index and searching."""
        index = BM25Index(k1=1.5, b=0.75)

        # Add documents
        index.add_document(1, "Python programming language")
        index.add_document(2, "Rust systems programming")
        index.add_document(3, "Python data science")

        # Build index
        index.build()

        # Search
        results = index.search("Python programming", k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score > results[1].score
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_empty_query(self):
        """Test with empty query."""
        index = BM25Index()
        index.add_document(1, "test document")
        index.build()

        results = index.search("", k=10)
        assert len(results) == 0

    def test_no_results(self):
        """Test when no documents match."""
        index = BM25Index()
        index.add_document(1, "Python programming")
        index.build()

        results = index.search("JavaScript", k=10)
        assert len(results) == 0

    def test_result_to_dict(self):
        """Test SearchResult.to_dict()."""
        index = BM25Index()
        index.add_document(1, "test document")
        index.build()

        results = index.search("test", k=1)
        assert len(results) == 1
        result_dict = results[0].to_dict()
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "rank" in result_dict

    def test_index_stats(self):
        """Test index statistics."""
        index = BM25Index()
        index.add_document(1, "the quick brown fox")
        index.add_document(2, "the lazy dog")
        index.build()

        stats = index.stats()
        assert stats["num_docs"] == 2
        assert stats["num_terms"] > 0
        assert stats["avg_doc_len"] > 0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSparseRetriever:
    """Test SparseRetriever with Rust backend."""

    def test_retriever_with_metadata(self, tmp_path):
        """Test retriever loading from metadata file."""
        from pocketwiki_chat.retrieval.sparse import SparseRetriever

        # Create metadata file
        metadata = {
            "k1": 1.5,
            "b": 0.75,
            "docs": [
                {"doc_id": 1, "text": "Python programming language"},
                {"doc_id": 2, "text": "Rust systems programming"},
                {"doc_id": 3, "text": "Python data science"},
            ],
        }

        index_dir = tmp_path / "bm25_index"
        index_dir.mkdir()
        metadata_path = index_dir / "bm25_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create retriever and search
        retriever = SparseRetriever(str(index_dir))
        results = retriever.search("Python programming", k=2)

        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk_1"
        assert results[0]["score"] > results[1]["score"]
        assert results[0]["rank"] == 0

    def test_retriever_missing_metadata(self, tmp_path):
        """Test error when metadata file is missing."""
        from pocketwiki_chat.retrieval.sparse import SparseRetriever

        index_dir = tmp_path / "empty_index"
        index_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="BM25 metadata not found"):
            SparseRetriever(str(index_dir))


@pytest.mark.skipif(RUST_AVAILABLE, reason="Test for when Rust is not available")
def test_sparse_retriever_without_rust():
    """Test that SparseRetriever raises error without Rust."""
    from pocketwiki_chat.retrieval.sparse import SparseRetriever
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="pocketwiki-rust not available"):
            SparseRetriever(tmpdir)
