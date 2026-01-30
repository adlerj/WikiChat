"""CLI for pocketwiki-builder."""
import time
from datetime import datetime
from pathlib import Path

import click

from pocketwiki_shared.schemas import (
    StreamParseConfig,
    ChunkConfig,
    FilterConfig,
    EmbedConfig,
    FAISSConfig,
    PackageConfig,
)
from .pipeline.stream_parse import StreamParseStage
from .pipeline.chunk import ChunkStage
from .pipeline.filter import FilterStage
from .pipeline.embed import EmbedStage
from .pipeline.faiss_index import FAISSIndexStage
from .pipeline.package import PackageStage


def _format_duration(seconds: float) -> str:
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


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory recursively."""
    total = 0
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


@click.group()
def cli():
    """PocketWiki Builder - Create Wikipedia bundles."""
    pass


@cli.command()
@click.option("--out", required=True, help="Output directory for bundle")
@click.option(
    "--source-url",
    default="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
    help="Wikipedia dump URL",
)
@click.option("--checkpoint-pages", default=1000, help="Pages between checkpoints")
@click.option("--max-chunk-tokens", default=512, help="Max tokens per chunk")
@click.option("--force-restart", is_flag=True, help="Force restart from beginning")
def build(
    out: str,
    source_url: str,
    checkpoint_pages: int,
    max_chunk_tokens: int,
    force_restart: bool,
):
    """Build a Wikipedia bundle."""
    pipeline_start = time.time()

    # Log pipeline start with all configuration
    print("=" * 70)
    print("POCKETWIKI BUILDER - Starting Pipeline")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {out}")
    print(f"  Source URL: {source_url}")
    print(f"  Checkpoint every: {checkpoint_pages} pages")
    print(f"  Max chunk tokens: {max_chunk_tokens}")
    print(f"  Force restart: {force_restart}")

    work_dir = Path(out) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWork directory: {work_dir}")
    print(f"  Created: {work_dir.exists()}")

    # Stage 1: StreamParse
    print("\n" + "=" * 70)
    print("STAGE 1/6: StreamParse")
    print("=" * 70)
    stream_config = StreamParseConfig(
        source_url=source_url,
        output_dir=str(work_dir / "parsed"),
        checkpoint_every_pages=checkpoint_pages,
        force_restart=force_restart,
    )
    stream_stage = StreamParseStage(stream_config, work_dir)
    stream_stage.execute()

    # Stage 2: Chunk
    print("\n" + "=" * 70)
    print("STAGE 2/6: Chunk")
    print("=" * 70)
    chunk_config = ChunkConfig(
        input_file=str(work_dir / "parsed" / "articles.jsonl"),
        output_dir=str(work_dir / "chunks"),
        max_chunk_tokens=max_chunk_tokens,
    )
    chunk_stage = ChunkStage(chunk_config, work_dir)
    chunk_stage.execute()

    # Stage 3: Filter
    print("\n" + "=" * 70)
    print("STAGE 3/6: Filter")
    print("=" * 70)
    filter_config = FilterConfig(
        input_file=str(work_dir / "chunks" / "chunks.jsonl"),
        output_dir=str(work_dir / "filtered"),
    )
    filter_stage = FilterStage(filter_config, work_dir)
    filter_stage.execute()

    # Stage 4: Embed
    print("\n" + "=" * 70)
    print("STAGE 4/6: Embed")
    print("=" * 70)
    embed_config = EmbedConfig(
        input_file=str(work_dir / "filtered" / "filtered.jsonl"),
        output_dir=str(work_dir / "embeddings"),
    )
    embed_stage = EmbedStage(embed_config, work_dir)
    embed_stage.execute()

    # Stage 5: FAISS Index
    print("\n" + "=" * 70)
    print("STAGE 5/6: FAISS Index")
    print("=" * 70)
    faiss_config = FAISSConfig(
        embeddings_file=str(work_dir / "embeddings" / "embeddings.npy"),
        output_dir=str(work_dir / "indexes"),
    )
    faiss_stage = FAISSIndexStage(faiss_config, work_dir)
    faiss_stage.execute()

    # Stage 6: Package
    print("\n" + "=" * 70)
    print("STAGE 6/6: Package")
    print("=" * 70)
    package_config = PackageConfig(
        work_dir=str(work_dir),
        output_bundle=str(Path(out) / "bundle"),
    )
    package_stage = PackageStage(package_config, work_dir)
    package_stage.execute()

    # Final summary
    pipeline_duration = time.time() - pipeline_start
    bundle_path = Path(out) / "bundle"
    bundle_size = _get_dir_size(bundle_path)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {_format_duration(pipeline_duration)}")
    print(f"\nBundle location: {bundle_path}")
    print(f"Bundle size: {_format_size(bundle_size)}")

    # List bundle contents
    if bundle_path.exists():
        print(f"\nBundle contents:")
        for f in sorted(bundle_path.iterdir()):
            if f.is_file():
                print(f"  {f.name}: {_format_size(f.stat().st_size)}")


if __name__ == "__main__":
    cli()
