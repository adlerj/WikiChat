"""Shared test fixtures."""
import json
import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return Path(__file__).parent


@pytest.fixture
def sample_wiki_xml(fixtures_dir: Path) -> Path:
    """Return path to sample Wikipedia XML file."""
    return fixtures_dir / "sample_wiki.xml"


@pytest.fixture
def sample_wiki_bz2(fixtures_dir: Path) -> Path:
    """Return path to bz2-compressed sample Wikipedia XML."""
    return fixtures_dir / "sample_wiki.xml.bz2"


@pytest.fixture
def temp_work_dir(tmp_path: Path) -> Path:
    """Create temporary work directory structure."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "parsed").mkdir()
    (work_dir / "checkpoints").mkdir()
    (work_dir / "chunks").mkdir()
    (work_dir / "embeddings").mkdir()
    (work_dir / "indexes").mkdir()
    return work_dir


@pytest.fixture
def mock_checkpoint_data() -> dict:
    """Return mock checkpoint data."""
    return {
        "source_url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        "source_etag": "abc123",
        "compressed_bytes_read": 1024,
        "pages_processed": 100,
        "last_page_id": "25433",
        "last_page_title": "Quantum mechanics",
        "output_file": "work/parsed/articles.jsonl",
        "output_bytes_written": 4096,
        "last_checkpoint_time": "2026-01-30T10:30:00Z",
        "checkpoint_version": 1,
    }


@pytest.fixture
def sample_articles() -> list:
    """Return sample article data."""
    return [
        {
            "id": "736",
            "title": "Albert Einstein",
            "text": "Albert Einstein was a theoretical physicist...",
        },
        {
            "id": "23862",
            "title": "Python (programming language)",
            "text": "Python is a high-level programming language...",
        },
        {
            "id": "25433",
            "title": "Quantum mechanics",
            "text": "Quantum mechanics is a fundamental theory...",
        },
    ]
