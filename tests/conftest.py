"""Root conftest for test fixtures."""
import sys
from pathlib import Path

# Add the tests directory to sys.path if not already there
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Re-export all fixtures from fixtures/conftest.py
from fixtures.conftest import *
