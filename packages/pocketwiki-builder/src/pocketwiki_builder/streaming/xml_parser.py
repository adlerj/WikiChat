"""Incremental XML parser for Wikipedia dumps."""
import re
from typing import Iterator, Dict, Optional, BinaryIO

from lxml import etree

from .errors import ParseError


class WikiXmlParser:
    """Incremental parser for Wikipedia XML dumps."""

    # Wikipedia MediaWiki namespace
    NS = "{http://www.mediawiki.org/xml/export-0.10/}"

    def __init__(
        self,
        skip_redirects: bool = True,
        skip_disambiguation: bool = False,
        allowed_namespaces: Optional[list[int]] = None,
    ):
        """Initialize parser.

        Args:
            skip_redirects: Skip redirect pages
            skip_disambiguation: Skip disambiguation pages
            allowed_namespaces: List of allowed namespace IDs (default: [0] for main)
        """
        self.skip_redirects = skip_redirects
        self.skip_disambiguation = skip_disambiguation
        self.allowed_namespaces = allowed_namespaces or [0]

    def parse(self, stream: BinaryIO) -> Iterator[Dict[str, str]]:
        """Parse Wikipedia XML stream incrementally.

        Args:
            stream: Binary stream of XML data

        Yields:
            Dictionary with keys: id, title, text, namespace
        """
        try:
            context = etree.iterparse(
                stream,
                events=("end",),
                tag=f"{self.NS}page",
            )

            for event, elem in context:
                try:
                    article = self._extract_article(elem)

                    if article and self._should_include(article):
                        yield article

                finally:
                    # Critical: Clear element to prevent memory leak
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]

        except etree.XMLSyntaxError as e:
            # Log but don't crash on malformed XML
            pass

    def _extract_article(self, elem: etree.Element) -> Optional[Dict[str, str]]:
        """Extract article data from page element.

        Args:
            elem: Page XML element

        Returns:
            Article dictionary or None if invalid
        """
        try:
            page_id = elem.findtext(f".//{self.NS}id")
            title = elem.findtext(f".//{self.NS}title")
            namespace = elem.findtext(f".//{self.NS}ns")
            text = elem.findtext(f".//{self.NS}revision/{self.NS}text")

            # Check for redirect
            redirect = elem.find(f".//{self.NS}redirect")
            is_redirect = redirect is not None

            if not all([page_id, title, text]):
                return None

            return {
                "id": page_id,
                "title": title,
                "text": text,
                "namespace": int(namespace) if namespace else 0,
                "is_redirect": is_redirect,
            }

        except Exception:
            return None

    def _should_include(self, article: Dict[str, str]) -> bool:
        """Check if article should be included.

        Args:
            article: Article dictionary

        Returns:
            True if article should be included
        """
        # Check namespace
        if article["namespace"] not in self.allowed_namespaces:
            return False

        # Check redirect
        if self.skip_redirects and article.get("is_redirect"):
            return False

        # Check if it's a redirect by text
        if self.skip_redirects and is_redirect(article["text"]):
            return False

        # Check disambiguation
        if self.skip_disambiguation and is_disambiguation(
            article["text"], article["title"]
        ):
            return False

        return True


def parse_wiki_xml_stream(byte_stream: Iterator[bytes]) -> Iterator[Dict[str, str]]:
    """Convenience function to parse Wikipedia XML from byte stream.

    Args:
        byte_stream: Iterator of byte chunks

    Yields:
        Article dictionaries
    """
    # Create a readable stream from iterator
    from io import BytesIO

    # Collect all bytes (for simplicity, could be optimized)
    data = b"".join(byte_stream)
    stream = BytesIO(data)

    parser = WikiXmlParser()
    yield from parser.parse(stream)


def is_redirect(text: str) -> bool:
    """Check if page text indicates a redirect.

    Args:
        text: Page text content

    Returns:
        True if page is a redirect
    """
    if not text:
        return False
    return text.strip().lower().startswith("#redirect")


def is_disambiguation(text: str, title: str = "") -> bool:
    """Check if page is a disambiguation page.

    Args:
        text: Page text content
        title: Page title

    Returns:
        True if page is disambiguation
    """
    if "(disambiguation)" in title:
        return True

    if not text:
        return False

    # Check for disambiguation templates
    disambig_patterns = [
        r"\{\{disambiguation\}\}",
        r"\{\{disambig\}\}",
        r"\{\{Disambiguation\}\}",
    ]

    text_lower = text.lower()
    for pattern in disambig_patterns:
        if re.search(pattern, text_lower):
            return True

    return False
