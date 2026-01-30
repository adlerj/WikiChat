"""Package stage - create final bundle."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

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
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        work_path = Path(self.config.work_dir)

        # Copy key files to bundle
        files_to_copy = [
            ("indexes/dense.faiss", "dense.faiss"),
            ("indexes/sparse.dict", "sparse.dict"),
            ("indexes/sparse.postings", "sparse.postings"),
            ("filtered/filtered.jsonl", "chunks.jsonl"),
        ]

        for src_rel, dst_name in files_to_copy:
            src = work_path / src_rel
            if src.exists():
                dst = self.bundle_dir / dst_name
                shutil.copy2(src, dst)

        # Create manifest
        manifest = {
            "version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_articles": 0,  # Would count from parsed
            "num_chunks": 0,  # Would count from chunks
        }

        manifest_file = self.bundle_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))

        print(f"Bundle created at {self.bundle_dir}")
