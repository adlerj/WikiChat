"""FAISS indexing stage."""
import hashlib
from pathlib import Path

import faiss
import numpy as np

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import FAISSConfig


class FAISSIndexStage(Stage):
    """Create FAISS IVF-PQ index."""

    def __init__(self, config: FAISSConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: FAISSConfig = config
        self.output_file = Path(config.output_dir) / "dense.faiss"

    def compute_input_hash(self) -> str:
        """Compute hash of config + input."""
        input_path = Path(self.config.embeddings_file)
        if input_path.exists():
            input_hash = hashlib.md5(
                str(input_path.stat().st_size).encode()
            ).hexdigest()[:8]
        else:
            input_hash = "none"
        config_hash = hashlib.sha256(
            self.config.model_dump_json().encode()
        ).hexdigest()[:8]
        return f"{input_hash}-{config_hash}"

    def get_output_files(self) -> list[Path]:
        return [self.output_file]

    def run(self) -> None:
        """Create FAISS index."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load embeddings
        embeddings = np.load(self.config.embeddings_file).astype("float32")
        n_vectors, dimension = embeddings.shape

        # Use simpler index for small datasets
        if n_vectors < self.config.n_clusters * 2:
            # Use flat index for small datasets (no training needed)
            print(f"Using flat index for {n_vectors} vectors (< {self.config.n_clusters * 2})")
            index = faiss.IndexFlatIP(dimension)
            # Normalize for inner product similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
        else:
            # Create IVF-PQ index for large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                self.config.n_clusters,
                self.config.n_subquantizers,
                self.config.bits_per_code,
            )

            # Train and add vectors
            print("Training FAISS index...")
            index.train(embeddings)
            index.add(embeddings)

        # Save index
        faiss.write_index(index, str(self.output_file))
        print(f"Created FAISS index with {index.ntotal} vectors")
