"""LLM generator (stub - requires llama-cpp-python)."""
from pathlib import Path
from typing import Iterator


class LLMGenerator:
    """LLM text generation (stub implementation)."""

    def __init__(self, model_path: Path):
        """Initialize generator."""
        self.model_path = model_path

    def generate(self, context: str, query: str) -> str:
        """Generate response.

        Args:
            context: Retrieved context
            query: User query

        Returns:
            Generated response
        """
        # Stub implementation
        return f"Response to: {query} (based on context)"

    def stream_generate(self, context: str, query: str) -> Iterator[str]:
        """Generate response with streaming.

        Args:
            context: Retrieved context
            query: User query

        Yields:
            Token strings
        """
        # Stub implementation
        response = self.generate(context, query)
        for word in response.split():
            yield word + " "
