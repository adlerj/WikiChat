"""StreamParse stage - streams and parses Wikipedia dumps with checkpointing."""
import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pocketwiki_shared.base import Stage
from pocketwiki_shared.schemas import StreamParseConfig, StreamParseCheckpoint

from ..streaming.checkpoint import CheckpointManager
from ..streaming.http_stream import stream_bz2_from_url, get_etag
from ..streaming.xml_parser import WikiXmlParser


class StreamParseStage(Stage):
    """Streaming Wikipedia dump parser with checkpoint support."""

    def __init__(self, config: StreamParseConfig, work_dir: Path):
        """Initialize stage.

        Args:
            config: StreamParse configuration
            work_dir: Working directory
        """
        super().__init__(config, work_dir)
        self.config: StreamParseConfig = config

        # Set up paths
        self.output_dir = Path(config.output_dir)
        self.output_file = self.output_dir / config.output_filename
        self.checkpoint_file = (
            work_dir / "checkpoints" / "stream_parse.checkpoint.json"
        )

        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_file, config)

    def compute_input_hash(self) -> str:
        """Compute hash of configuration."""
        return hashlib.sha256(
            self.config.model_dump_json().encode()
        ).hexdigest()[:16]

    def get_output_files(self) -> list[Path]:
        """Get list of output files."""
        return [self.output_file]

    def run(self) -> None:
        """Execute streaming parse with checkpoint support."""
        # Check if we should resume
        if self.config.force_restart:
            checkpoint = None
        else:
            checkpoint = self.checkpoint_mgr.load_checkpoint()

        if checkpoint and self._should_resume_from_checkpoint():
            self._resume_parse()
        else:
            self._fresh_parse()

    def _should_resume_from_checkpoint(self) -> bool:
        """Check if we should resume from checkpoint.

        Returns:
            True if checkpoint is valid and can be resumed
        """
        if not self.checkpoint_mgr.is_checkpoint_valid():
            return False

        checkpoint = self.checkpoint_mgr.load_checkpoint()
        if checkpoint is None:
            return False

        # Check if output file exists and matches checkpoint
        if not self.output_file.exists():
            return False

        # Verify output file size matches
        actual_size = self.output_file.stat().st_size
        if actual_size != checkpoint.output_bytes_written:
            return False

        return True

    def _fresh_parse(self) -> None:
        """Start fresh parse from beginning."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get source ETag for validation
        source_etag = None
        if self.config.validate_source_unchanged:
            try:
                source_etag = get_etag(str(self.config.source_url))
            except Exception:
                pass

        # Open output file
        with open(self.output_file, "w") as out_file:
            # Stream from URL
            byte_stream = stream_bz2_from_url(
                str(self.config.source_url),
                start_byte=0,
                chunk_size=self.config.http_chunk_size,
                max_retries=self.config.max_retries,
                timeout=self.config.http_timeout,
            )

            # Parse XML
            parser = WikiXmlParser(
                skip_redirects=self.config.skip_redirects,
                skip_disambiguation=self.config.skip_disambiguation,
                allowed_namespaces=self.config.allowed_namespaces,
            )

            # Collect bytes for parsing
            xml_bytes = b"".join(byte_stream)
            xml_stream = BytesIO(xml_bytes)
            compressed_bytes_read = len(xml_bytes)

            # Parse and write articles with progress
            self._parse_and_write(
                parser,
                xml_stream,
                out_file,
                compressed_bytes_read,
                source_etag,
                pages_processed=0,
                bytes_written=0,
            )

    def _resume_parse(self) -> None:
        """Resume parse from checkpoint."""
        checkpoint = self.checkpoint_mgr.load_checkpoint()
        if checkpoint is None:
            # Fallback to fresh parse
            self._fresh_parse()
            return

        print(f"Resuming from checkpoint: {checkpoint.pages_processed} pages processed")

        # Open output file in append mode
        with open(self.output_file, "a") as out_file:
            # Stream from URL with Range request
            byte_stream = stream_bz2_from_url(
                str(self.config.source_url),
                start_byte=checkpoint.compressed_bytes_read,
                chunk_size=self.config.http_chunk_size,
                max_retries=self.config.max_retries,
                timeout=self.config.http_timeout,
            )

            # Parse XML
            parser = WikiXmlParser(
                skip_redirects=self.config.skip_redirects,
                skip_disambiguation=self.config.skip_disambiguation,
                allowed_namespaces=self.config.allowed_namespaces,
            )

            # Collect bytes
            xml_bytes = b"".join(byte_stream)
            xml_stream = BytesIO(xml_bytes)

            # Parse and write, continuing from checkpoint
            self._parse_and_write(
                parser,
                xml_stream,
                out_file,
                checkpoint.compressed_bytes_read + len(xml_bytes),
                checkpoint.source_etag,
                pages_processed=checkpoint.pages_processed,
                bytes_written=checkpoint.output_bytes_written,
            )

    def _parse_and_write(
        self,
        parser: WikiXmlParser,
        xml_stream: BytesIO,
        out_file,
        compressed_bytes_read: int,
        source_etag: Optional[str],
        pages_processed: int,
        bytes_written: int,
    ) -> None:
        """Parse XML and write articles with checkpointing.

        Args:
            parser: XML parser
            xml_stream: XML byte stream
            out_file: Output file handle
            compressed_bytes_read: Total compressed bytes read
            source_etag: Source ETag for validation
            pages_processed: Pages processed so far
            bytes_written: Bytes written so far
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "Streaming Wikipedia dump...", total=None
            )

            for article in parser.parse(xml_stream):
                # Write article as JSON line
                line = json.dumps(article) + "\n"
                out_file.write(line)
                bytes_written += len(line.encode("utf-8"))
                pages_processed += 1

                # Update progress
                progress.update(
                    task,
                    description=f"Parsed {pages_processed:,} pages",
                )

                # Check if we should checkpoint
                if self.checkpoint_mgr.should_checkpoint(
                    pages_processed, bytes_written
                ):
                    checkpoint = StreamParseCheckpoint(
                        source_url=str(self.config.source_url),
                        source_etag=source_etag,
                        compressed_bytes_read=compressed_bytes_read,
                        pages_processed=pages_processed,
                        last_page_id=article.get("id"),
                        last_page_title=article.get("title"),
                        output_file=str(self.output_file),
                        output_bytes_written=bytes_written,
                        last_checkpoint_time=datetime.utcnow().isoformat(),
                    )
                    self.checkpoint_mgr.save_checkpoint(checkpoint)
                    self.checkpoint_mgr.reset_counters()

            # Final checkpoint
            checkpoint = StreamParseCheckpoint(
                source_url=str(self.config.source_url),
                source_etag=source_etag,
                compressed_bytes_read=compressed_bytes_read,
                pages_processed=pages_processed,
                last_page_id=article.get("id") if article else None,
                last_page_title=article.get("title") if article else None,
                output_file=str(self.output_file),
                output_bytes_written=bytes_written,
                last_checkpoint_time=datetime.utcnow().isoformat(),
            )
            self.checkpoint_mgr.save_checkpoint(checkpoint)

            print(f"\n✓ Parsed {pages_processed:,} pages total")
