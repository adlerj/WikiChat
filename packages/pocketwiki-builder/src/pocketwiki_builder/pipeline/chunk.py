"""Chunking stage - split articles into smaller chunks."""
import hashlib
import json
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import ChunkConfig


class ChunkStage(Stage):
    """Split articles into token-sized chunks."""

    def __init__(self, config: ChunkConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: ChunkConfig = config
        self.output_file = Path(config.output_dir) / "chunks.jsonl"

    def compute_input_hash(self) -> str:
        """Compute hash of config + input file."""
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
        """Chunk articles."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Log input file info
        input_path = Path(self.config.input_file)
        if input_path.exists():
            input_size = input_path.stat().st_size
            print(f"\n  Input file: {input_path}")
            print(f"  Input size: {input_size:,} bytes")
        else:
            print(f"\n  WARNING: Input file not found: {input_path}")

        print(f"  Chunking parameters:")
        print(f"    Max chunk tokens: {self.config.max_chunk_tokens}")
        print(f"    Overlap tokens: {self.config.overlap_tokens}")

        article_count = 0
        total_chunks = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Chunking articles...", total=None)

            with open(self.config.input_file, "r") as in_file:
                with open(self.output_file, "w") as out_file:
                    for line in in_file:
                        article = json.loads(line)
                        article_count += 1

                        # Simple chunking: split by tokens (approximated as words)
                        text = article["text"]
                        words = text.split()
                        chunk_size = self.config.max_chunk_tokens

                        chunks = [
                            " ".join(words[i : i + chunk_size])
                            for i in range(0, len(words), chunk_size)
                        ]

                        for i, chunk_text in enumerate(chunks):
                            chunk = {
                                "chunk_id": f"{article['id']}-{i}",
                                "page_id": article["id"],
                                "page_title": article["title"],
                                "text": chunk_text,
                                "chunk_index": i,
                            }
                            out_file.write(json.dumps(chunk) + "\n")
                            total_chunks += 1

                        # Update progress
                        progress.update(
                            task,
                            description=f"Chunked {article_count:,} articles → {total_chunks:,} chunks",
                        )

        print(f"\n  Results:")
        print(f"    Articles processed: {article_count:,}")
        print(f"    Total chunks created: {total_chunks:,}")
        print(f"    Avg chunks per article: {total_chunks / max(article_count, 1):.1f}")
