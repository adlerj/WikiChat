"""FAISS indexing stage."""
import hashlib
from pathlib import Path

import faiss
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
        print(f"\n  Loading embeddings from: {self.config.embeddings_file}")
        embeddings = np.load(self.config.embeddings_file).astype("float32")
        n_vectors, dimension = embeddings.shape
        print(f"  Loaded {n_vectors:,} vectors of dimension {dimension}")

        # Use simpler index for small datasets
        threshold = self.config.n_clusters * 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            if n_vectors < threshold:
                # Use flat index for small datasets (no training needed)
                print(f"\n  Index selection: FLAT (exact search)")
                print(f"    Reason: {n_vectors:,} vectors < threshold of {threshold:,}")
                print(f"    Index type: IndexFlatIP (inner product)")
                index = faiss.IndexFlatIP(dimension)

                # Normalize for inner product similarity
                task = progress.add_task("Normalizing vectors...", total=None)
                faiss.normalize_L2(embeddings)
                progress.update(task, completed=True)

                # Add vectors to index
                task = progress.add_task(f"Adding {n_vectors:,} vectors to index...", total=None)
                index.add(embeddings)
                progress.update(task, completed=True)
            else:
                # Create IVF-PQ index for large datasets
                print(f"\n  Index selection: IVF-PQ (approximate search)")
                print(f"    Reason: {n_vectors:,} vectors >= threshold of {threshold:,}")
                print(f"    Parameters:")
                print(f"      n_clusters: {self.config.n_clusters}")
                print(f"      n_subquantizers: {self.config.n_subquantizers}")
                print(f"      bits_per_code: {self.config.bits_per_code}")

                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFPQ(
                    quantizer,
                    dimension,
                    self.config.n_clusters,
                    self.config.n_subquantizers,
                    self.config.bits_per_code,
                )

                # Train and add vectors
                task = progress.add_task("Training FAISS index...", total=None)
                index.train(embeddings)
                progress.update(task, completed=True)

                task = progress.add_task(f"Adding {n_vectors:,} vectors to index...", total=None)
                index.add(embeddings)
                progress.update(task, completed=True)

        # Save index
        faiss.write_index(index, str(self.output_file))

        print(f"\n  Results:")
        print(f"    Total vectors in index: {index.ntotal:,}")
        print(f"    Index file: {self.output_file}")
        if self.output_file.exists():
            size = self.output_file.stat().st_size
            print(f"    Index file size: {size:,} bytes")
