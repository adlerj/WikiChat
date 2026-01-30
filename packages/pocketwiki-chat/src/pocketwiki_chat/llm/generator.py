"""LLM generator using llama-cpp-python."""
from pathlib import Path
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class LLMGenerator:
    """LLM text generation using llama-cpp-python.

    Supports GGUF model files with streaming generation.
    Model is loaded lazily on first use.
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """Initialize generator.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            verbose: Enable verbose llama.cpp output
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model: Optional["Llama"] = None

    @property
    def model(self) -> "Llama":
        """Get or load the LLM model (lazy loading)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> "Llama":
        """Load the GGUF model.

        Returns:
            Loaded Llama model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        from llama_cpp import Llama

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        logger.info(f"Loading LLM model from {self.model_path}")

        try:
            model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
            logger.info("LLM model loaded successfully")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def generate(
        self,
        context: str,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate response (non-streaming).

        Args:
            context: Retrieved context from RAG
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stop: Optional stop sequences

        Returns:
            Generated response text
        """
        from pocketwiki_chat.llm.prompts import format_rag_prompt

        prompt = format_rag_prompt(query=query, context=context)

        if stop is None:
            stop = ["Question:", "\n\nContext:"]

        response = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
        )

        return response["choices"][0]["text"].strip()

    def stream_generate(
        self,
        context: str,
        query: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Generate response with streaming.

        Args:
            context: Retrieved context from RAG
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stop: Optional stop sequences

        Yields:
            Token strings as they are generated
        """
        from pocketwiki_chat.llm.prompts import format_rag_prompt

        prompt = format_rag_prompt(query=query, context=context)

        if stop is None:
            stop = ["Question:", "\n\nContext:"]

        stream = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
            stream=True,
        )

        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("LLM model unloaded")
