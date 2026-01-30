"""Dense retrieval with FAISS."""
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense vector search with FAISS."""

    def __init__(self, index_path: Path, model_name: str):
        """Initialize retriever.

        Args:
            index_path: Path to FAISS index
            model_name: Sentence-transformers model name
        """
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for similar chunks.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of results with chunk_id, score, rank
        """
        # Encode query
        query_vec = self.model.encode([query])[0].astype("float32")
        query_vec = np.expand_dims(query_vec, axis=0)

        # Search
        distances, indices = self.index.search(query_vec, k)

        # Format results
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            results.append({
                "chunk_id": str(idx),
                "score": float(dist),
                "rank": rank,
            })

        return results
