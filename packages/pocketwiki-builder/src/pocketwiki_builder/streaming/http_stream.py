"""HTTP streaming with bz2 decompression."""
import bz2
import time
from typing import Iterator, Optional

import requests
from requests.exceptions import HTTPError, Timeout, RequestException

from .errors import HttpStreamError


def stream_bz2_from_url(
    url: str,
    start_byte: int = 0,
    chunk_size: int = 1024 * 1024,  # 1 MB
    max_retries: int = 5,
    timeout: int = 300,
) -> Iterator[bytes]:
    """Stream bz2-compressed data from URL with resume support.

    Args:
        url: URL to stream from
        start_byte: Byte offset to resume from
        chunk_size: Size of chunks to read
        max_retries: Maximum number of retries
        timeout: Request timeout in seconds

    Yields:
        Decompressed byte chunks

    Raises:
        HttpStreamError: If streaming fails after retries
    """
    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    retries = 0
    backoff = 10  # seconds

    while retries <= max_retries:
        try:
            response = requests.get(
                url,
                stream=True,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            # Initialize decompressor
            decompressor = bz2.BZ2Decompressor()

            # Stream and decompress
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    try:
                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            yield decompressed
                    except EOFError:
                        # End of bz2 stream
                        break

            # Success, return
            return

        except (HTTPError, Timeout, RequestException) as e:
            # Check if it's a 4xx error (don't retry)
            if isinstance(e, HTTPError) and 400 <= e.response.status_code < 500:
                raise HttpStreamError(f"HTTP {e.response.status_code}: {e}") from e

            retries += 1
            if retries > max_retries:
                raise HttpStreamError(
                    f"Failed after {max_retries} retries: {e}"
                ) from e

            # Exponential backoff
            sleep_time = backoff * (2 ** (retries - 1))
            time.sleep(sleep_time)


def get_etag(url: str) -> Optional[str]:
    """Get ETag header from URL.

    Args:
        url: URL to check

    Returns:
        ETag value or None if not present

    Raises:
        HttpStreamError: If request fails
    """
    try:
        response = requests.head(url, timeout=30)
        response.raise_for_status()
        return response.headers.get("ETag")
    except (HTTPError, RequestException) as e:
        raise HttpStreamError(f"Failed to get ETag: {e}") from e


def supports_range_requests(url: str) -> bool:
    """Check if server supports Range requests.

    Args:
        url: URL to check

    Returns:
        True if Range requests supported
    """
    try:
        response = requests.head(url, timeout=30)
        response.raise_for_status()
        accept_ranges = response.headers.get("Accept-Ranges", "")
        return accept_ranges.lower() == "bytes"
    except (HTTPError, RequestException):
        return False
