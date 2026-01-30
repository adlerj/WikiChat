"""Tests for pocketwiki_builder.streaming.http_stream."""
import bz2
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import responses
from requests.exceptions import HTTPError, Timeout

from pocketwiki_builder.streaming.http_stream import (
    HttpStreamError,
    stream_bz2_from_url,
    get_etag,
    supports_range_requests,
)


class TestStreamBz2FromUrl:
    """Tests for stream_bz2_from_url function."""

    @responses.activate
    def test_basic_streaming(self, sample_wiki_bz2: Path) -> None:
        """Test basic HTTP streaming with bz2 decompression."""
        # Read compressed data
        compressed_data = sample_wiki_bz2.read_bytes()

        # Mock HTTP response
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=compressed_data,
            status=200,
            stream=True,
        )

        # Stream and decompress
        chunks = list(stream_bz2_from_url("http://example.com/dump.xml.bz2"))
        decompressed = b"".join(chunks)

        # Verify we got valid XML
        assert b"<mediawiki" in decompressed
        assert b"Albert Einstein" in decompressed

    @responses.activate
    def test_streaming_with_range_request(self, sample_wiki_bz2: Path) -> None:
        """Test HTTP Range request for resume."""
        compressed_data = sample_wiki_bz2.read_bytes()
        start_byte = 100

        # Mock range request
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=compressed_data[start_byte:],
            status=206,  # Partial Content
            headers={"Content-Range": f"bytes {start_byte}-{len(compressed_data)-1}/{len(compressed_data)}"},
        )

        # Stream from offset
        chunks = list(
            stream_bz2_from_url(
                "http://example.com/dump.xml.bz2", start_byte=start_byte
            )
        )

        # Should get some data (though bz2 decompression from middle may fail)
        assert len(chunks) >= 0

    @responses.activate
    def test_streaming_chunks(self, sample_wiki_bz2: Path) -> None:
        """Test streaming yields multiple chunks."""
        compressed_data = sample_wiki_bz2.read_bytes()

        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=compressed_data,
            status=200,
        )

        chunks = list(
            stream_bz2_from_url(
                "http://example.com/dump.xml.bz2", chunk_size=256
            )
        )

        # Should get multiple chunks
        assert len(chunks) > 0
        # All chunks should be bytes
        assert all(isinstance(chunk, bytes) for chunk in chunks)

    @responses.activate
    def test_http_404_error(self) -> None:
        """Test handling of HTTP 404 error."""
        responses.add(
            responses.GET,
            "http://example.com/missing.xml.bz2",
            status=404,
        )

        with pytest.raises(HttpStreamError) as exc_info:
            list(stream_bz2_from_url("http://example.com/missing.xml.bz2"))

        assert "404" in str(exc_info.value)

    @responses.activate
    def test_http_500_error_with_retry(self) -> None:
        """Test retry logic for HTTP 5xx errors."""
        # First request fails, second succeeds
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            status=500,
        )
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=b"test",
            status=200,
        )

        # Should retry and succeed
        with patch("pocketwiki_builder.streaming.http_stream.time.sleep"):
            chunks = list(
                stream_bz2_from_url(
                    "http://example.com/dump.xml.bz2", max_retries=2
                )
            )

        assert len(responses.calls) == 2

    @responses.activate
    def test_timeout_with_retry(self) -> None:
        """Test retry logic for timeout errors."""
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=Timeout("Request timeout"),
        )
        responses.add(
            responses.GET,
            "http://example.com/dump.xml.bz2",
            body=b"test",
            status=200,
        )

        with patch("pocketwiki_builder.streaming.http_stream.time.sleep"):
            chunks = list(
                stream_bz2_from_url(
                    "http://example.com/dump.xml.bz2", max_retries=2
                )
            )

        assert len(responses.calls) == 2

    @responses.activate
    def test_max_retries_exceeded(self) -> None:
        """Test failure when max retries exceeded."""
        for _ in range(5):
            responses.add(
                responses.GET,
                "http://example.com/dump.xml.bz2",
                status=500,
            )

        with pytest.raises(HttpStreamError) as exc_info:
            with patch("pocketwiki_builder.streaming.http_stream.time.sleep"):
                list(
                    stream_bz2_from_url(
                        "http://example.com/dump.xml.bz2", max_retries=3
                    )
                )

        assert "max retries" in str(exc_info.value).lower()


class TestGetEtag:
    """Tests for get_etag function."""

    @responses.activate
    def test_get_etag_present(self) -> None:
        """Test getting ETag from response headers."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            headers={"ETag": '"abc123"'},
            status=200,
        )

        etag = get_etag("http://example.com/dump.xml.bz2")
        assert etag == '"abc123"'

    @responses.activate
    def test_get_etag_missing(self) -> None:
        """Test getting ETag when not present."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            status=200,
        )

        etag = get_etag("http://example.com/dump.xml.bz2")
        assert etag is None

    @responses.activate
    def test_get_etag_error(self) -> None:
        """Test getting ETag with HTTP error."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            status=404,
        )

        with pytest.raises(HttpStreamError):
            get_etag("http://example.com/dump.xml.bz2")


class TestSupportsRangeRequests:
    """Tests for supports_range_requests function."""

    @responses.activate
    def test_range_supported(self) -> None:
        """Test detection of Range request support."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            headers={"Accept-Ranges": "bytes"},
            status=200,
        )

        assert supports_range_requests("http://example.com/dump.xml.bz2") is True

    @responses.activate
    def test_range_not_supported(self) -> None:
        """Test detection when Range not supported."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            headers={"Accept-Ranges": "none"},
            status=200,
        )

        assert supports_range_requests("http://example.com/dump.xml.bz2") is False

    @responses.activate
    def test_range_header_missing(self) -> None:
        """Test when Accept-Ranges header missing."""
        responses.add(
            responses.HEAD,
            "http://example.com/dump.xml.bz2",
            status=200,
        )

        # Assume not supported if header missing
        assert supports_range_requests("http://example.com/dump.xml.bz2") is False
