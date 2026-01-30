"""Tests for web API."""
from fastapi.testclient import TestClient
import pytest


class TestChatAPI:
    """Tests for chat API endpoints."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        from pocketwiki_chat.web.app import create_app

        app = create_app()
        return TestClient(app)

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root serves HTML."""
        response = client.get("/")
        assert response.status_code == 200

    def test_chat_endpoint(self, client: TestClient) -> None:
        """Test chat endpoint."""
        response = client.post(
            "/api/chat",
            json={"query": "What is Python?"},
        )
        assert response.status_code == 200

    def test_search_endpoint(self, client: TestClient) -> None:
        """Test search-only endpoint."""
        response = client.post(
            "/api/search",
            json={"query": "Einstein"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_page_endpoint(self, client: TestClient) -> None:
        """Test fetching full page."""
        response = client.get("/api/page/736")
        assert response.status_code in [200, 404]  # May not exist in test

    def test_sse_streaming(self, client: TestClient) -> None:
        """Test streaming parameter accepted."""
        response = client.post(
            "/api/chat",
            json={"query": "test", "stream": True},
        )
        assert response.status_code == 200
        # Stub implementation returns JSON, full implementation would use SSE


class TestLLMIntegration:
    """Tests for LLM integration."""

    @pytest.mark.skip(reason="Requires LLM model file")
    def test_llm_generate(self) -> None:
        """Test LLM generation."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        generator = LLMGenerator(model_path="path/to/model.gguf")
        response = generator.generate(
            context="Test context",
            query="Test query",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_rag_prompt_template(self) -> None:
        """Test RAG prompt formatting."""
        from pocketwiki_chat.llm.prompts import format_rag_prompt

        prompt = format_rag_prompt(
            query="What is Python?",
            context="Python is a programming language.",
        )

        assert "Python" in prompt
        assert "programming language" in prompt
