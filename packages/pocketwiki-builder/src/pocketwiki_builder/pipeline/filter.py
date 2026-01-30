"""Filtering stage - remove low-quality chunks."""
import hashlib
import json
from pathlib import Path

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import FilterConfig


class FilterStage(Stage):
    """Filter low-quality chunks."""

    def __init__(self, config: FilterConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: FilterConfig = config
        self.output_file = Path(config.output_dir) / "filtered.jsonl"

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
        """Filter chunks."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.input_file, "r") as in_file:
            with open(self.output_file, "w") as out_file:
                for line in in_file:
                    chunk = json.loads(line)

                    # Filter by length
                    text_len = len(chunk["text"])
                    if (
                        text_len < self.config.min_chunk_length
                        or text_len > self.config.max_chunk_length
                    ):
                        continue

                    out_file.write(json.dumps(chunk) + "\n")
