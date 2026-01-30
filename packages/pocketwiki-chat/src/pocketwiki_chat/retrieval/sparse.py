"""Sparse retrieval with BM25."""
from typing import List, Dict


class SparseRetriever:
    """BM25 sparse retrieval (stub - would use Rust implementation)."""

    def __init__(self, index_dir: str):
        """Initialize retriever."""
        self.index_dir = index_dir

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search with BM25.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of results with chunk_id, score, rank
        """
        # Stub implementation
        return []
