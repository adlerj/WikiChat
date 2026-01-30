"""Bundle loader for chat app."""
import json
from pathlib import Path
from typing import Optional, Dict, Any


class BundleLoader:
    """Load and validate Wikipedia bundles."""

    def __init__(self, bundle_dir: Path):
        """Initialize loader.

        Args:
            bundle_dir: Path to bundle directory
        """
        self.bundle_dir = Path(bundle_dir)
        self.manifest_file = self.bundle_dir / "manifest.json"

    def load_manifest(self) -> Dict[str, Any]:
        """Load bundle manifest.

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest doesn't exist
        """
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")

        return json.loads(self.manifest_file.read_text())

    def validate(self) -> bool:
        """Validate bundle has required files.

        Returns:
            True if bundle is valid
        """
        required_files = [
            "manifest.json",
            "dense.faiss",
        ]

        for filename in required_files:
            if not (self.bundle_dir / filename).exists():
                return False

        return True

    def get_dense_index_path(self) -> Path:
        """Get path to dense FAISS index."""
        return self.bundle_dir / "dense.faiss"

    def get_chunks_path(self) -> Path:
        """Get path to chunks file."""
        return self.bundle_dir / "chunks.jsonl"
