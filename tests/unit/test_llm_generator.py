"""Tests for LLM generator."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


class TestLLMGenerator:
    """Tests for LLMGenerator class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test generator initialization."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        generator = LLMGenerator(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=10,
            verbose=True,
        )

        assert generator.model_path == model_path
        assert generator.n_ctx == 2048
        assert generator.n_gpu_layers == 10
        assert generator.verbose is True
        assert generator._model is None

    def test_init_defaults(self, tmp_path: Path) -> None:
        """Test generator initialization with defaults."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        generator = LLMGenerator(model_path=model_path)

        assert generator.n_ctx == 4096
        assert generator.n_gpu_layers == 0
        assert generator.verbose is False

    def test_is_loaded_false(self, tmp_path: Path) -> None:
        """Test is_loaded returns False before loading."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        generator = LLMGenerator(model_path=tmp_path / "model.gguf")
        assert generator.is_loaded() is False

    def test_load_model_file_not_found(self, tmp_path: Path) -> None:
        """Test loading raises FileNotFoundError for missing model."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        generator = LLMGenerator(model_path=tmp_path / "missing.gguf")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            generator.model

    @patch("llama_cpp.Llama", autospec=False)
    def test_load_model_success(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test successful model loading."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=5,
            verbose=True,
        )

        result = generator.model

        assert result is mock_model
        mock_llama_class.assert_called_once_with(
            model_path=str(model_path),
            n_ctx=2048,
            n_gpu_layers=5,
            verbose=True,
        )

    @patch("llama_cpp.Llama", autospec=False)
    def test_load_model_lazy(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test model is loaded lazily."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        generator = LLMGenerator(model_path=model_path)

        # Model not loaded yet
        mock_llama_class.assert_not_called()
        assert generator._model is None

        # Access triggers load
        _ = generator.model
        mock_llama_class.assert_called_once()

    @patch("llama_cpp.Llama", autospec=False)
    def test_load_model_cached(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test model is cached after first load."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        generator = LLMGenerator(model_path=model_path)

        _ = generator.model
        _ = generator.model
        _ = generator.model

        # Only called once
        mock_llama_class.assert_called_once()

    @patch("llama_cpp.Llama", autospec=False)
    def test_load_model_failure(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test loading raises RuntimeError on failure."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_llama_class.side_effect = Exception("Load failed")

        generator = LLMGenerator(model_path=model_path)

        with pytest.raises(RuntimeError, match="Failed to load model"):
            generator.model

    @patch("llama_cpp.Llama", autospec=False)
    def test_generate(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test non-streaming generation."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_model.create_completion.return_value = {
            "choices": [{"text": "  Generated response text  "}]
        }
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(model_path=model_path)
        result = generator.generate(
            context="Test context",
            query="Test query",
            max_tokens=256,
            temperature=0.5,
        )

        assert result == "Generated response text"
        mock_model.create_completion.assert_called_once()
        call_kwargs = mock_model.create_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs.get("stream", False) is False  # Default is False
        assert "Test query" in call_kwargs["prompt"]
        assert "Test context" in call_kwargs["prompt"]

    @patch("llama_cpp.Llama", autospec=False)
    def test_generate_default_stop_sequences(
        self, mock_llama_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test default stop sequences are used."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_model.create_completion.return_value = {"choices": [{"text": "response"}]}
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(model_path=model_path)
        generator.generate(context="ctx", query="q")

        call_kwargs = mock_model.create_completion.call_args[1]
        assert "Question:" in call_kwargs["stop"]
        assert "\n\nContext:" in call_kwargs["stop"]

    @patch("llama_cpp.Llama", autospec=False)
    def test_generate_custom_stop_sequences(
        self, mock_llama_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test custom stop sequences override defaults."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_model.create_completion.return_value = {"choices": [{"text": "response"}]}
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(model_path=model_path)
        generator.generate(context="ctx", query="q", stop=["<stop>"])

        call_kwargs = mock_model.create_completion.call_args[1]
        assert call_kwargs["stop"] == ["<stop>"]

    @patch("llama_cpp.Llama", autospec=False)
    def test_stream_generate(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test streaming generation."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
            {"choices": [{"text": "!"}]},
        ])
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(model_path=model_path)
        tokens = list(generator.stream_generate(context="ctx", query="q"))

        assert tokens == ["Hello", " world", "!"]
        call_kwargs = mock_model.create_completion.call_args[1]
        assert call_kwargs["stream"] is True

    @patch("llama_cpp.Llama", autospec=False)
    def test_stream_generate_skips_empty(
        self, mock_llama_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test streaming skips empty tokens."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        mock_model = MagicMock()
        mock_model.create_completion.return_value = iter([
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": ""}]},  # Empty
            {"choices": [{"text": " world"}]},
        ])
        mock_llama_class.return_value = mock_model

        generator = LLMGenerator(model_path=model_path)
        tokens = list(generator.stream_generate(context="ctx", query="q"))

        assert tokens == ["Hello", " world"]

    @patch("llama_cpp.Llama", autospec=False)
    def test_is_loaded_true(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test is_loaded returns True after loading."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        generator = LLMGenerator(model_path=model_path)
        _ = generator.model

        assert generator.is_loaded() is True

    @patch("llama_cpp.Llama", autospec=False)
    def test_unload(self, mock_llama_class: MagicMock, tmp_path: Path) -> None:
        """Test unloading model."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"fake gguf data")

        generator = LLMGenerator(model_path=model_path)
        _ = generator.model
        assert generator.is_loaded() is True

        generator.unload()
        assert generator.is_loaded() is False
        assert generator._model is None

    def test_unload_when_not_loaded(self, tmp_path: Path) -> None:
        """Test unload is safe when not loaded."""
        from pocketwiki_chat.llm.generator import LLMGenerator

        generator = LLMGenerator(model_path=tmp_path / "model.gguf")
        generator.unload()  # Should not raise
        assert generator.is_loaded() is False
