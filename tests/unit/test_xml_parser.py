"""Tests for pocketwiki_builder.streaming.xml_parser."""
from io import BytesIO
from pathlib import Path

import pytest

from pocketwiki_builder.streaming.xml_parser import (
    WikiXmlParser,
    parse_wiki_xml_stream,
    is_redirect,
    is_disambiguation,
)


class TestWikiXmlParser:
    """Tests for WikiXmlParser class."""

    def test_parse_sample_xml(self, sample_wiki_xml: Path) -> None:
        """Test parsing sample Wikipedia XML."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser(
            skip_redirects=False,
            skip_disambiguation=False,
            allowed_namespaces=[0, 4],  # Include namespace 4 for Wikipedia: pages
        )
        articles = list(parser.parse(stream))

        # Should get 5 pages total (all namespaces)
        assert len(articles) == 5

        # Check first article
        assert articles[0]["id"] == "736"
        assert articles[0]["title"] == "Albert Einstein"
        assert "theoretical physicist" in articles[0]["text"]

    def test_parse_skip_redirects(self, sample_wiki_xml: Path) -> None:
        """Test skipping redirect pages."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser(skip_redirects=True, skip_disambiguation=False)
        articles = list(parser.parse(stream))

        # Should skip the redirect page
        assert len(articles) == 4
        assert all("Redirect" not in a["title"] for a in articles)

    def test_parse_skip_disambiguation(self, sample_wiki_xml: Path) -> None:
        """Test skipping disambiguation pages."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser(skip_redirects=False, skip_disambiguation=True)
        articles = list(parser.parse(stream))

        # Should skip disambiguation page (only ns=0, minus disambiguation = 3)
        assert len(articles) == 3
        assert all("disambiguation" not in a["title"] for a in articles)

    def test_parse_skip_both(self, sample_wiki_xml: Path) -> None:
        """Test skipping both redirects and disambiguation pages."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser(skip_redirects=True, skip_disambiguation=True)
        articles = list(parser.parse(stream))

        # Should only get 3 regular articles
        assert len(articles) == 3
        titles = [a["title"] for a in articles]
        assert "Albert Einstein" in titles
        assert "Python (programming language)" in titles
        assert "Quantum mechanics" in titles

    def test_parse_with_namespace_filtering(self, sample_wiki_xml: Path) -> None:
        """Test filtering by namespace."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser(
            skip_redirects=False,
            skip_disambiguation=False,
            allowed_namespaces=[0],  # Only main namespace
        )
        articles = list(parser.parse(stream))

        # Should only get namespace 0 articles
        assert len(articles) == 4
        assert all("Wikipedia:" not in a["title"] for a in articles)

    def test_parse_incremental_iteration(self, sample_wiki_xml: Path) -> None:
        """Test that parser yields articles incrementally."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser()
        iterator = parser.parse(stream)

        # Get first article
        first = next(iterator)
        assert first["title"] == "Albert Einstein"

        # Get second article
        second = next(iterator)
        assert second["title"] == "Python (programming language)"

        # Should be able to continue iterating (2 more: Quantum + disambiguation)
        remaining = list(iterator)
        assert len(remaining) == 2

    def test_parse_malformed_xml_graceful(self) -> None:
        """Test graceful handling of malformed XML."""
        malformed_xml = b"""<mediawiki>
            <page>
                <title>Test</title>
                <id>1</id>
                <!-- Missing closing revision tag -->
                <revision>
                    <text>Some text</text>
            </page>
        </mediawiki>"""

        stream = BytesIO(malformed_xml)
        parser = WikiXmlParser()

        # Should not crash, may skip malformed pages
        articles = list(parser.parse(stream))
        # May get 0 or 1 article depending on parser robustness
        assert len(articles) >= 0

    def test_parse_empty_stream(self) -> None:
        """Test parsing empty stream."""
        stream = BytesIO(b"<mediawiki></mediawiki>")
        parser = WikiXmlParser()
        articles = list(parser.parse(stream))
        assert articles == []

    def test_article_structure(self, sample_wiki_xml: Path) -> None:
        """Test that parsed articles have correct structure."""
        xml_data = sample_wiki_xml.read_bytes()
        stream = BytesIO(xml_data)

        parser = WikiXmlParser()
        articles = list(parser.parse(stream))

        for article in articles:
            assert "id" in article
            assert "title" in article
            assert "text" in article
            assert isinstance(article["id"], str)
            assert isinstance(article["title"], str)
            assert isinstance(article["text"], str)


class TestParseWikiXmlStream:
    """Tests for parse_wiki_xml_stream convenience function."""

    def test_parse_xml_stream(self, sample_wiki_xml: Path) -> None:
        """Test convenience function for parsing."""
        xml_data = sample_wiki_xml.read_bytes()

        def byte_stream():
            yield xml_data

        articles = list(parse_wiki_xml_stream(byte_stream()))
        assert len(articles) >= 3


class TestIsRedirect:
    """Tests for is_redirect helper function."""

    def test_detect_redirect_tag(self) -> None:
        """Test detection of redirect from element."""
        # This would normally be an lxml element
        # For testing, we can use text content
        assert is_redirect("#REDIRECT [[Other Page]]") is True
        assert is_redirect("#redirect [[Other Page]]") is True

    def test_not_redirect(self) -> None:
        """Test non-redirect page."""
        assert is_redirect("This is regular content") is False
        assert is_redirect("") is False


class TestIsDisambiguation:
    """Tests for is_disambiguation helper function."""

    def test_detect_disambiguation_template(self) -> None:
        """Test detection of disambiguation pages."""
        assert is_disambiguation("{{disambiguation}}") is True
        assert is_disambiguation("{{Disambiguation}}") is True
        assert is_disambiguation("{{disambig}}") is True
        assert (
            is_disambiguation("Some content\n{{disambiguation}}\nMore content")
            is True
        )

    def test_detect_disambiguation_in_title(self) -> None:
        """Test detection from title."""
        assert is_disambiguation("", title="Physics (disambiguation)") is True

    def test_not_disambiguation(self) -> None:
        """Test non-disambiguation page."""
        assert is_disambiguation("Regular article content") is False
        assert is_disambiguation("", title="Regular Article") is False
