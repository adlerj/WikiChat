"""Sparse retrieval with BM25."""
from typing import List, Dict
from pathlib import Path
import json

try:
    from pocketwiki_rust import BM25Index
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class SparseRetriever:
    """BM25 sparse retrieval using Rust implementation."""

    def __init__(self, index_dir: str):
        """Initialize retriever.

        Args:
            index_dir: Directory containing BM25 index files
        """
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "pocketwiki-rust not available. Install with: "
                "pip install /path/to/pocketwiki_rust-*.whl"
            )

        self.index_dir = Path(index_dir)
        self.index = self._load_index()

    def _load_index(self) -> BM25Index:
        """Load BM25 index from disk.

        Expected file structure:
        - index_dir/bm25_metadata.json: {docs: [{doc_id, text}, ...], k1, b}
        """
        metadata_path = self.index_dir / "bm25_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"BM25 metadata not found at {metadata_path}. "
                "Run builder pipeline with SparseIndex stage first."
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create index with parameters
        k1 = metadata.get("k1", 1.5)
        b = metadata.get("b", 0.75)
        index = BM25Index(k1=k1, b=b)

        # Add documents
        for doc in metadata["docs"]:
            index.add_document(doc["doc_id"], doc["text"])

        # Build compressed postings
        index.build()

        return index

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search with BM25.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of results with chunk_id, score, rank
        """
        results = self.index.search(query, k)
        return [
            {
                "chunk_id": result.chunk_id,
                "score": result.score,
                "rank": result.rank,
            }
            for result in results
        ]
