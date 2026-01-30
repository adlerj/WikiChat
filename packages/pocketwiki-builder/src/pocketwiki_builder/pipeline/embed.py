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

        # Load model
        model = SentenceTransformer(self.config.model_name)

        # Read chunks
        chunks = []
        with open(self.config.input_file, "r") as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk["text"])

        # Generate embeddings in batches
        embeddings = model.encode(
            chunks,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
        )

        # Save embeddings
        np.save(self.output_file, embeddings)
        print(f"Generated {len(embeddings)} embeddings")
