"""Filtering stage - remove low-quality chunks."""
import hashlib
import json
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

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

        # Log filter criteria
        print(f"\n  Filter criteria:")
        print(f"    Min chunk length: {self.config.min_chunk_length} chars")
        print(f"    Max chunk length: {self.config.max_chunk_length} chars")

        total_input = 0
        kept = 0
        too_short = 0
        too_long = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Filtering chunks...", total=None)

            with open(self.config.input_file, "r") as in_file:
                with open(self.output_file, "w") as out_file:
                    for line in in_file:
                        chunk = json.loads(line)
                        total_input += 1

                        # Filter by length
                        text_len = len(chunk["text"])
                        if text_len < self.config.min_chunk_length:
                            too_short += 1
                        elif text_len > self.config.max_chunk_length:
                            too_long += 1
                        else:
                            kept += 1
                            out_file.write(json.dumps(chunk) + "\n")

                        # Update progress
                        if total_input % 100 == 0:
                            filtered_out = total_input - kept
                            progress.update(
                                task,
                                description=f"Processed {total_input:,} chunks → kept {kept:,} ({100*kept/max(total_input,1):.1f}%)",
                            )

        filtered_out = total_input - kept
        print(f"\n  Results:")
        print(f"    Input chunks: {total_input:,}")
        print(f"    Kept chunks: {kept:,}")
        print(f"    Filtered out: {filtered_out:,} ({100*filtered_out/max(total_input,1):.1f}%)")
        print(f"      - Too short (<{self.config.min_chunk_length}): {too_short:,}")
        print(f"      - Too long (>{self.config.max_chunk_length}): {too_long:,}")
