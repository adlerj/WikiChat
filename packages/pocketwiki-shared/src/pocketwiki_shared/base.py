"""Base Stage class for pipeline."""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
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
        self._start_time: Optional[float] = None

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
        return self.work_dir / f"{self.get_stage_name()}.state.json"

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
            completed_at=datetime.now(timezone.utc).isoformat(),
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

    def _log_config_summary(self) -> None:
        """Log configuration summary for this stage."""
        stage_name = self.get_stage_name()
        config_dict = self.config.model_dump()
        print(f"  Config:")
        for key, value in config_dict.items():
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 60:
                str_val = str_val[:57] + "..."
            print(f"    {key}: {str_val}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def execute(self) -> None:
        """Execute stage if needed, handling skip logic."""
        stage_name = self.get_stage_name()

        # Log stage start with config
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}")
        print(f"{'='*60}")
        self._log_config_summary()

        # Compute and log input hash
        input_hash = self.compute_input_hash()
        print(f"  Input hash: {input_hash}")

        # Check skip logic with detailed logging
        state = self.load_state()
        if state is not None:
            print(f"  Previous state found:")
            print(f"    Completed: {state.completed}")
            print(f"    Previous hash: {state.input_hash}")
            print(f"    Hash match: {state.input_hash == input_hash}")

        if self.should_skip():
            print(f"\n→ SKIPPING: Stage already completed with matching inputs")
            if state:
                print(f"  Completed at: {state.completed_at}")
            return

        # Log why we're running
        if state is None:
            print(f"\n→ RUNNING: No previous state found")
        elif not state.completed:
            print(f"\n→ RUNNING: Previous run was incomplete")
        elif state.input_hash != input_hash:
            print(f"\n→ RUNNING: Input hash changed")
        else:
            missing = [f for f in self.get_output_files() if not f.exists()]
            print(f"\n→ RUNNING: Output files missing: {missing}")

        # Execute with timing
        self._start_time = time.time()
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.run()

        # Log completion
        duration = time.time() - self._start_time
        self.persist_state()

        print(f"\n✓ {stage_name} COMPLETED")
        print(f"  Duration: {self._format_duration(duration)}")
        print(f"  Output files:")
        for output_file in self.get_output_files():
            if output_file.exists():
                size = output_file.stat().st_size
                size_str = self._format_size(size)
                print(f"    {output_file.name}: {size_str}")
            else:
                print(f"    {output_file.name}: (not created)")

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"
