"""Base Stage class for pipeline."""
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import StageState, StageConfig


class Stage(ABC):
    """Base class for pipeline stages."""

    def __init__(self, config: StageConfig, work_dir: Path):
        """Initialize stage.

        Args:
            config: Stage configuration
            work_dir: Working directory for all pipeline artifacts
        """
        self.config = config
        self.work_dir = Path(work_dir)

    @abstractmethod
    def compute_input_hash(self) -> str:
        """Compute hash of inputs to detect changes.

        Returns:
            Hex string hash of inputs
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Execute the stage logic."""
        pass

    def get_stage_name(self) -> str:
        """Get stage name from class name.

        Returns:
            Snake-case stage name
        """
        name = self.__class__.__name__
        # Convert CamelCase to snake_case
        import re

        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def get_state_file(self) -> Path:
        """Get path to state file.

        Returns:
            Path to {stage_name}.state.json
        """
        return self.work_dir / "state" / f"{self.get_stage_name()}.state.json"

    def get_output_files(self) -> list[Path]:
        """Get list of output files produced by this stage.

        Returns:
            List of output file paths
        """
        return []

    def load_state(self) -> Optional[StageState]:
        """Load persisted state if exists.

        Returns:
            StageState if exists, None otherwise
        """
        state_file = self.get_state_file()
        if not state_file.exists():
            return None

        try:
            return StageState.model_validate_json(state_file.read_text())
        except Exception:
            return None

    def persist_state(self) -> None:
        """Persist stage completion state."""
        state = StageState(
            stage_name=self.get_stage_name(),
            input_hash=self.compute_input_hash(),
            completed=True,
            completed_at=datetime.utcnow().isoformat(),
            output_files=[str(f) for f in self.get_output_files()],
        )

        state_file = self.get_state_file()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(state.model_dump_json(indent=2))

    def should_skip(self) -> bool:
        """Check if stage should be skipped (already completed with same inputs).

        Returns:
            True if stage can be skipped, False otherwise
        """
        state = self.load_state()
        if state is None:
            return False

        if not state.completed:
            return False

        # Check if input hash matches
        current_hash = self.compute_input_hash()
        if state.input_hash != current_hash:
            return False

        # Check if output files exist
        for output_file in self.get_output_files():
            if not output_file.exists():
                return False

        return True

    def execute(self) -> None:
        """Execute stage if needed, handling skip logic."""
        if self.should_skip():
            print(f"✓ Skipping {self.get_stage_name()} (already completed)")
            return

        print(f"→ Running {self.get_stage_name()}...")
        self.run()
        self.persist_state()
        print(f"✓ {self.get_stage_name()} completed")
