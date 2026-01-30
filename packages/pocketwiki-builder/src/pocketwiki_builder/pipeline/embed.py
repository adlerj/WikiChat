"""Embedding stage - generate embeddings for chunks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import EmbedConfig


class EmbedStage(Stage):
    """Generate embeddings for chunks."""

    def __init__(self, config: EmbedConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: EmbedConfig = config
        self.output_file = Path(config.output_dir) / "embeddings.npy"

    def compute_input_hash(self) -> str:
        """Compute hash of config + input."""
        input_path = Path(self.config.input_file)
        if input_path.exists():
            input_hash = hashlib.md5(input_path.read_bytes()).hexdigest()[:8]
        else:
            input_hash = "none"
        config_hash = hashlib.sha256(
            self.config.model_dump_json().encode()
        ).hexdigest()[:8]
        return f"{input_hash}-{config_hash}"

    def get_output_files(self) -> list[Path]:
        return [self.output_file]

    def run(self) -> None:
        """Generate embeddings."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Log model loading
        print(f"\n  Loading embedding model: {self.config.model_name}")
        print(f"  Batch size: {self.config.batch_size}")
        model = SentenceTransformer(self.config.model_name)
        print(f"  Model loaded successfully")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

        # Read chunks
        print(f"\n  Reading chunks from: {self.config.input_file}")
        chunks = []
        with open(self.config.input_file, "r") as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk["text"])
        print(f"  Loaded {len(chunks):,} chunks")

        # Generate embeddings in batches
        print(f"\n  Generating embeddings...")
        num_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size
        print(f"  Total batches: {num_batches}")
        embeddings = model.encode(
            chunks,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
        )

        # Save embeddings
        np.save(self.output_file, embeddings)
        print(f"\n  Results:")
        print(f"    Generated {len(embeddings):,} embeddings")
        print(f"    Embedding shape: {embeddings.shape}")
        print(f"    Output file: {self.output_file}")
