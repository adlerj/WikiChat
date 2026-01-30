"""CLI for pocketwiki-builder."""
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
    work_dir = Path(out) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: StreamParse
    print("\n=== Stage 1: StreamParse ===")
    stream_config = StreamParseConfig(
        source_url=source_url,
        output_dir=str(work_dir / "parsed"),
        checkpoint_every_pages=checkpoint_pages,
        force_restart=force_restart,
    )
    stream_stage = StreamParseStage(stream_config, work_dir)
    stream_stage.execute()

    # Stage 2: Chunk
    print("\n=== Stage 2: Chunk ===")
    chunk_config = ChunkConfig(
        input_file=str(work_dir / "parsed" / "articles.jsonl"),
        output_dir=str(work_dir / "chunks"),
        max_chunk_tokens=max_chunk_tokens,
    )
    chunk_stage = ChunkStage(chunk_config, work_dir)
    chunk_stage.execute()

    # Stage 3: Filter
    print("\n=== Stage 3: Filter ===")
    filter_config = FilterConfig(
        input_file=str(work_dir / "chunks" / "chunks.jsonl"),
        output_dir=str(work_dir / "filtered"),
    )
    filter_stage = FilterStage(filter_config, work_dir)
    filter_stage.execute()

    # Stage 4: Embed
    print("\n=== Stage 4: Embed ===")
    embed_config = EmbedConfig(
        input_file=str(work_dir / "filtered" / "filtered.jsonl"),
        output_dir=str(work_dir / "embeddings"),
    )
    embed_stage = EmbedStage(embed_config, work_dir)
    embed_stage.execute()

    # Stage 5: FAISS Index
    print("\n=== Stage 5: FAISS Index ===")
    faiss_config = FAISSConfig(
        embeddings_file=str(work_dir / "embeddings" / "embeddings.npy"),
        output_dir=str(work_dir / "indexes"),
    )
    faiss_stage = FAISSIndexStage(faiss_config, work_dir)
    faiss_stage.execute()

    # Stage 6: Package
    print("\n=== Stage 6: Package ===")
    package_config = PackageConfig(
        work_dir=str(work_dir),
        output_bundle=str(Path(out) / "bundle"),
    )
    package_stage = PackageStage(package_config, work_dir)
    package_stage.execute()

    print(f"\n✓ Build complete! Bundle at {Path(out) / 'bundle'}")


if __name__ == "__main__":
    cli()
