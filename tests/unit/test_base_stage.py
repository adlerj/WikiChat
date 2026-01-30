"""Tests for pocketwiki_shared.base Stage class."""
import hashlib
from pathlib import Path
from typing import Any

import pytest

from pocketwiki_shared.base import Stage, StageConfig
from pocketwiki_shared.schemas import StageState


class MockConfig(StageConfig):
    """Mock configuration for testing."""

    value: int = 10


class MockStage(Stage):
    """Mock stage implementation for testing."""

    def __init__(self, config: MockConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.run_called = False

    def compute_input_hash(self) -> str:
        """Compute hash of configuration."""
        return hashlib.sha256(
            self.config.model_dump_json().encode()
        ).hexdigest()[:16]

    def run(self) -> None:
        """Execute the stage."""
        self.run_called = True
        output_file = self.work_dir / "output.txt"
        output_file.write_text("test output")

    def get_output_files(self) -> list[Path]:
        """Return list of output files."""
        return [self.work_dir / "output.txt"]


class TestStageBase:
    """Tests for Stage base class."""

    def test_stage_should_skip_no_state(self, temp_work_dir: Path) -> None:
        """Test should_skip returns False when no state file exists."""
        config = MockConfig()
        stage = MockStage(config, temp_work_dir)
        assert stage.should_skip() is False

    def test_stage_should_skip_with_matching_state(
        self, temp_work_dir: Path
    ) -> None:
        """Test should_skip returns True when state matches."""
        config = MockConfig(value=10)
        stage = MockStage(config, temp_work_dir)

        # Create output file and state
        stage.run()
        stage.persist_state()

        # Create new stage instance with same config
        stage2 = MockStage(config, temp_work_dir)
        assert stage2.should_skip() is True

    def test_stage_should_skip_with_changed_config(
        self, temp_work_dir: Path
    ) -> None:
        """Test should_skip returns False when config changes."""
        config = MockConfig(value=10)
        stage = MockStage(config, temp_work_dir)
        stage.run()
        stage.persist_state()

        # Create new stage with different config
        config2 = MockConfig(value=20)
        stage2 = MockStage(config2, temp_work_dir)
        assert stage2.should_skip() is False

    def test_stage_persist_state(self, temp_work_dir: Path) -> None:
        """Test persist_state writes state file."""
        config = MockConfig()
        stage = MockStage(config, temp_work_dir)
        stage.run()

        state_file = temp_work_dir / "mock_stage.state.json"
        assert not state_file.exists()

        stage.persist_state()
        assert state_file.exists()

        # Verify state content
        state = StageState.model_validate_json(state_file.read_text())
        assert state.completed is True
        assert state.stage_name == "mock_stage"

    def test_stage_load_state(self, temp_work_dir: Path) -> None:
        """Test load_state reads state file correctly."""
        config = MockConfig()
        stage = MockStage(config, temp_work_dir)
        stage.run()
        stage.persist_state()

        # Load state in new instance
        stage2 = MockStage(config, temp_work_dir)
        state = stage2.load_state()
        assert state is not None
        assert state.completed is True

    def test_stage_execute_runs_when_needed(self, temp_work_dir: Path) -> None:
        """Test execute() runs stage when not skipped."""
        config = MockConfig()
        stage = MockStage(config, temp_work_dir)

        stage.execute()

        assert stage.run_called is True
        assert (temp_work_dir / "output.txt").exists()

    def test_stage_execute_skips_when_completed(
        self, temp_work_dir: Path
    ) -> None:
        """Test execute() skips when already completed."""
        config = MockConfig()
        stage = MockStage(config, temp_work_dir)
        stage.execute()

        # Second execution should skip
        stage2 = MockStage(config, temp_work_dir)
        stage2.execute()

        # run_called should still be False for second instance
        assert stage2.run_called is False
