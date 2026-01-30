"""Package stage - create final bundle."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import PackageConfig


class PackageStage(Stage):
    """Package everything into final bundle."""

    def __init__(self, config: PackageConfig, work_dir: Path):
        super().__init__(config, work_dir)
        self.config: PackageConfig = config
        self.bundle_dir = Path(config.output_bundle)

    def compute_input_hash(self) -> str:
        """Compute hash of config."""
        return hashlib.sha256(
            self.config.model_dump_json().encode()
        ).hexdigest()[:16]

    def get_output_files(self) -> list[Path]:
        return [self.bundle_dir / "manifest.json"]

    def run(self) -> None:
        """Create bundle."""
        print(f"\n  Creating bundle at: {self.bundle_dir}")
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        work_path = Path(self.config.work_dir)

        # Copy key files to bundle
        files_to_copy = [
            ("indexes/dense.faiss", "dense.faiss"),
            ("indexes/sparse.dict", "sparse.dict"),
            ("indexes/sparse.postings", "sparse.postings"),
            ("filtered/filtered.jsonl", "chunks.jsonl"),
        ]

        print(f"\n  Copying files to bundle:")
        total_size = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            copy_task = progress.add_task("Copying files...", total=len(files_to_copy))

            for src_rel, dst_name in files_to_copy:
                src = work_path / src_rel
                if src.exists():
                    dst = self.bundle_dir / dst_name
                    shutil.copy2(src, dst)
                    size = src.stat().st_size
                    total_size += size
                    print(f"    {src_rel} -> {dst_name} ({size:,} bytes)")
                else:
                    print(f"    {src_rel} -> SKIPPED (not found)")

                progress.advance(copy_task)

            # Count chunks for manifest
            num_chunks = 0
            num_articles = set()
            chunks_file = self.bundle_dir / "chunks.jsonl"
            if chunks_file.exists():
                count_task = progress.add_task("Counting chunks...", total=None)
                with open(chunks_file, "r") as f:
                    for line in f:
                        num_chunks += 1
                        chunk = json.loads(line)
                        num_articles.add(chunk.get("page_id"))
                        if num_chunks % 1000 == 0:
                            progress.update(count_task, description=f"Counted {num_chunks:,} chunks...")
                progress.update(count_task, completed=True)

        # Create manifest
        manifest = {
            "version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_articles": len(num_articles),
            "num_chunks": num_chunks,
        }

        manifest_file = self.bundle_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print(f"    manifest.json created")

        print(f"\n  Manifest contents:")
        print(f"    version: {manifest['version']}")
        print(f"    num_articles: {manifest['num_articles']:,}")
        print(f"    num_chunks: {manifest['num_chunks']:,}")

        print(f"\n  Bundle summary:")
        print(f"    Location: {self.bundle_dir}")
        print(f"    Total size: {total_size:,} bytes")
