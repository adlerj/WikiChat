"""End-to-end integration tests for PocketWikiRAG."""
import bz2
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# Sample Wikipedia XML for testing
SAMPLE_XML = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <siteinfo>
    <sitename>Test Wikipedia</sitename>
    <dbname>testwiki</dbname>
  </siteinfo>
  <page>
    <title>Albert Einstein</title>
    <ns>0</ns>
    <id>736</id>
    <revision>
      <id>1001</id>
      <timestamp>2024-01-01T00:00:00Z</timestamp>
      <text>Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.</text>
    </revision>
  </page>
  <page>
    <title>Python (programming language)</title>
    <ns>0</ns>
    <id>23862</id>
    <revision>
      <id>2001</id>
      <timestamp>2024-01-02T00:00:00Z</timestamp>
      <text>Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. Python was conceived in the late 1980s by Guido van Rossum.</text>
    </revision>
  </page>
  <page>
    <title>Machine learning</title>
    <ns>0</ns>
    <id>233488</id>
    <revision>
      <id>3001</id>
      <timestamp>2024-01-03T00:00:00Z</timestamp>
      <text>Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data. Recently, artificial neural networks have been able to surpass many previous approaches in performance. Machine learning approaches have been applied to natural language processing, computer vision, and speech recognition.</text>
    </revision>
  </page>
  <page>
    <title>Redirect to Einstein</title>
    <ns>0</ns>
    <id>88888</id>
    <revision>
      <id>5001</id>
      <timestamp>2024-01-05T00:00:00Z</timestamp>
      <text>#REDIRECT [[Albert Einstein]]</text>
    </revision>
  </page>
</mediawiki>"""


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def work_dir(self) -> Path:
        """Create temporary work directory."""
        tmp = tempfile.mkdtemp(prefix="pocketwiki_integration_")
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_xml_parsing(self, work_dir: Path) -> None:
        """Test XML parsing produces valid articles."""
        from io import BytesIO
        from pocketwiki_builder.streaming.xml_parser import WikiXmlParser

        # Parse XML
        parser = WikiXmlParser(skip_redirects=True, skip_disambiguation=True)
        xml_bytes = SAMPLE_XML.encode("utf-8")
        articles = list(parser.parse(BytesIO(xml_bytes)))

        # Should have 3 articles (redirect and disambiguation skipped)
        assert len(articles) == 3

        # Check article content
        titles = [a["title"] for a in articles]
        assert "Albert Einstein" in titles
        assert "Python (programming language)" in titles
        assert "Machine learning" in titles

        # Redirect should be skipped
        assert "Redirect to Einstein" not in titles

    def test_chunking(self, work_dir: Path) -> None:
        """Test article chunking."""
        from pocketwiki_builder.pipeline.chunk import ChunkStage
        from pocketwiki_shared.schemas import ChunkConfig

        # Create input file
        parsed_dir = work_dir / "parsed"
        parsed_dir.mkdir()
        input_file = parsed_dir / "articles.jsonl"

        with open(input_file, "w") as f:
            f.write(json.dumps({
                "id": "736",
                "title": "Albert Einstein",
                "text": "Albert Einstein was a physicist. " * 100,  # Long text
                "namespace": 0,
            }) + "\n")

        # Run chunking
        chunk_config = ChunkConfig(
            input_file=str(input_file),
            output_dir=str(work_dir / "chunks"),
            max_chunk_tokens=50,
        )
        stage = ChunkStage(chunk_config, work_dir)
        stage.run()

        # Check output
        output_file = work_dir / "chunks" / "chunks.jsonl"
        assert output_file.exists()

        chunks = []
        with open(output_file) as f:
            for line in f:
                chunks.append(json.loads(line))

        # Should have multiple chunks due to small max_chunk_tokens
        assert len(chunks) >= 2
        assert all(c["page_title"] == "Albert Einstein" for c in chunks)

    def test_filtering(self, work_dir: Path) -> None:
        """Test chunk filtering."""
        from pocketwiki_builder.pipeline.filter import FilterStage
        from pocketwiki_shared.schemas import FilterConfig

        # Create input file with varying quality chunks
        chunks_dir = work_dir / "chunks"
        chunks_dir.mkdir()
        input_file = chunks_dir / "chunks.jsonl"

        with open(input_file, "w") as f:
            # Good chunk with many words
            f.write(json.dumps({
                "chunk_id": "1",
                "page_id": "736",
                "page_title": "Einstein",
                "text": "Albert Einstein was a renowned physicist who developed the theory of relativity and made many important contributions to science.",
            }) + "\n")
            # Short chunk (should be filtered)
            f.write(json.dumps({
                "chunk_id": "2",
                "page_id": "736",
                "page_title": "Einstein",
                "text": "Short text here.",
            }) + "\n")

        # Run filtering with higher min_tokens to ensure filtering happens
        filter_config = FilterConfig(
            input_file=str(input_file),
            output_dir=str(work_dir / "filtered"),
            min_tokens=15,  # First chunk has ~20 words, second has ~3
        )
        stage = FilterStage(filter_config, work_dir)
        stage.run()

        # Check output
        output_file = work_dir / "filtered" / "filtered.jsonl"
        assert output_file.exists()

        filtered = []
        with open(output_file) as f:
            for line in f:
                filtered.append(json.loads(line))

        # Short chunk should be filtered out
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "1"

    def test_embedding(self, work_dir: Path) -> None:
        """Test embedding generation."""
        from pocketwiki_builder.pipeline.embed import EmbedStage
        from pocketwiki_shared.schemas import EmbedConfig

        # Create input file
        filtered_dir = work_dir / "filtered"
        filtered_dir.mkdir()
        input_file = filtered_dir / "filtered.jsonl"

        with open(input_file, "w") as f:
            f.write(json.dumps({
                "chunk_id": "1",
                "text": "Albert Einstein was a physicist.",
            }) + "\n")
            f.write(json.dumps({
                "chunk_id": "2",
                "text": "Python is a programming language.",
            }) + "\n")

        # Run embedding
        embed_config = EmbedConfig(
            input_file=str(input_file),
            output_dir=str(work_dir / "embeddings"),
        )
        stage = EmbedStage(embed_config, work_dir)
        stage.run()

        # Check output
        output_file = work_dir / "embeddings" / "embeddings.npy"
        assert output_file.exists()

        embeddings = np.load(output_file)
        assert embeddings.shape[0] == 2  # 2 chunks
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_faiss_indexing(self, work_dir: Path) -> None:
        """Test FAISS index creation."""
        import faiss
        from pocketwiki_builder.pipeline.faiss_index import FAISSIndexStage
        from pocketwiki_shared.schemas import FAISSConfig

        # Create embeddings
        embeddings_dir = work_dir / "embeddings"
        embeddings_dir.mkdir()
        embeddings = np.random.randn(10, 384).astype("float32")
        np.save(embeddings_dir / "embeddings.npy", embeddings)

        # Run indexing
        faiss_config = FAISSConfig(
            embeddings_file=str(embeddings_dir / "embeddings.npy"),
            output_dir=str(work_dir / "indexes"),
        )
        stage = FAISSIndexStage(faiss_config, work_dir)
        stage.run()

        # Check output
        index_file = work_dir / "indexes" / "dense.faiss"
        assert index_file.exists()

        index = faiss.read_index(str(index_file))
        assert index.ntotal == 10

    def test_full_pipeline(self, work_dir: Path) -> None:
        """Test full pipeline from parsing to bundle."""
        from io import BytesIO
        from pocketwiki_builder.streaming.xml_parser import WikiXmlParser
        from pocketwiki_builder.pipeline.chunk import ChunkStage
        from pocketwiki_builder.pipeline.filter import FilterStage
        from pocketwiki_builder.pipeline.embed import EmbedStage
        from pocketwiki_builder.pipeline.faiss_index import FAISSIndexStage
        from pocketwiki_builder.pipeline.package import PackageStage
        from pocketwiki_shared.schemas import (
            ChunkConfig, FilterConfig, EmbedConfig, FAISSConfig, PackageConfig
        )

        # Stage 1: Parse XML
        parser = WikiXmlParser(skip_redirects=True, skip_disambiguation=True)
        xml_bytes = SAMPLE_XML.encode("utf-8")
        articles = list(parser.parse(BytesIO(xml_bytes)))

        parsed_dir = work_dir / "parsed"
        parsed_dir.mkdir()
        with open(parsed_dir / "articles.jsonl", "w") as f:
            for article in articles:
                f.write(json.dumps(article) + "\n")

        # Stage 2: Chunk
        chunk_config = ChunkConfig(
            input_file=str(parsed_dir / "articles.jsonl"),
            output_dir=str(work_dir / "chunks"),
            max_chunk_tokens=200,
        )
        ChunkStage(chunk_config, work_dir).run()

        # Stage 3: Filter
        filter_config = FilterConfig(
            input_file=str(work_dir / "chunks" / "chunks.jsonl"),
            output_dir=str(work_dir / "filtered"),
            min_tokens=10,
        )
        FilterStage(filter_config, work_dir).run()

        # Stage 4: Embed
        embed_config = EmbedConfig(
            input_file=str(work_dir / "filtered" / "filtered.jsonl"),
            output_dir=str(work_dir / "embeddings"),
        )
        EmbedStage(embed_config, work_dir).run()

        # Stage 5: FAISS Index
        faiss_config = FAISSConfig(
            embeddings_file=str(work_dir / "embeddings" / "embeddings.npy"),
            output_dir=str(work_dir / "indexes"),
        )
        FAISSIndexStage(faiss_config, work_dir).run()

        # Stage 6: Package
        package_config = PackageConfig(
            work_dir=str(work_dir),
            output_bundle=str(work_dir / "bundle"),
        )
        PackageStage(package_config, work_dir).run()

        # Verify bundle
        bundle_dir = work_dir / "bundle"
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "dense.faiss").exists()
        assert (bundle_dir / "chunks.jsonl").exists()

        # Load and verify manifest
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        assert manifest["version"] == "0.1.0"

    def test_chat_with_bundle(self, work_dir: Path) -> None:
        """Test chat app with created bundle."""
        import faiss
        from pocketwiki_chat.bundle.loader import BundleLoader
        from pocketwiki_chat.retrieval.dense import DenseRetriever
        from pocketwiki_chat.retrieval.fusion import rrf_fusion

        # Create minimal bundle
        bundle_dir = work_dir / "bundle"
        bundle_dir.mkdir()

        # Create chunks file
        chunks = [
            {"chunk_id": "0", "page_id": "736", "page_title": "Einstein", "text": "Einstein was a physicist."},
            {"chunk_id": "1", "page_id": "23862", "page_title": "Python", "text": "Python is a programming language."},
        ]
        with open(bundle_dir / "chunks.jsonl", "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        # Create FAISS index with embeddings
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts).astype("float32")

        index = faiss.IndexFlatIP(384)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(bundle_dir / "dense.faiss"))

        # Create manifest
        manifest = {"version": "0.1.0", "created_at": "2024-01-01T00:00:00Z"}
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

        # Test bundle loading
        loader = BundleLoader(bundle_dir)
        assert loader.validate()

        # Test dense retrieval
        retriever = DenseRetriever(bundle_dir / "dense.faiss")
        results = retriever.search("physicist scientist", k=2)
        assert len(results) == 2

        # Einstein should rank higher for physics query
        chunk_ids = [r["chunk_id"] for r in results]
        assert "0" in chunk_ids  # Einstein chunk

        # Test RRF fusion
        dense_results = [{"chunk_id": "0", "rank": 0}, {"chunk_id": "1", "rank": 1}]
        sparse_results = [{"chunk_id": "1", "rank": 0}, {"chunk_id": "0", "rank": 1}]
        fused = rrf_fusion(dense_results, sparse_results)
        assert len(fused) == 2


class TestRustBM25Integration:
    """Test Rust BM25 integration."""

    @pytest.fixture
    def work_dir(self) -> Path:
        """Create temporary work directory."""
        tmp = tempfile.mkdtemp(prefix="pocketwiki_bm25_")
        yield Path(tmp)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_bm25_index_creation(self) -> None:
        """Test BM25 index creation with Rust backend."""
        try:
            from pocketwiki_rust import BM25Index
        except ImportError:
            pytest.skip("Rust extension not available")

        index = BM25Index()
        # Note: BM25Index uses string chunk IDs internally (prefixed with "chunk_")
        index.add_document(0, "Albert Einstein was a physicist")
        index.add_document(1, "Python is a programming language")
        index.add_document(2, "Machine learning uses neural networks")
        index.build()

        # Search for physics
        results = index.search("physicist scientist", k=3)
        assert len(results) >= 1

        # Einstein should be first (chunk_id is string "chunk_0")
        assert results[0].chunk_id == "chunk_0"

    def test_sparse_retriever(self, work_dir: Path) -> None:
        """Test SparseRetriever with Rust backend."""
        try:
            from pocketwiki_rust import BM25Index
        except ImportError:
            pytest.skip("Rust extension not available")

        # Create BM25 metadata file with correct schema (doc_id, not chunk_id)
        metadata = {
            "docs": [
                {"doc_id": 0, "text": "Einstein was a physicist"},
                {"doc_id": 1, "text": "Python programming language"},
            ],
            "k1": 1.2,
            "b": 0.75,
        }
        (work_dir / "bm25_metadata.json").write_text(json.dumps(metadata))

        # Test sparse retriever
        from pocketwiki_chat.retrieval.sparse import SparseRetriever
        retriever = SparseRetriever(work_dir)

        results = retriever.search("physicist", k=2)
        assert len(results) >= 1
