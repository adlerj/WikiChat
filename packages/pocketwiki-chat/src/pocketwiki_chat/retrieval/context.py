"""Context assembly from chunks."""
from typing import List, Dict


def assemble_context(chunks: List[Dict], max_tokens: int = 4000) -> str:
    """Assemble context from chunks.

    Args:
        chunks: List of chunk dictionaries with text, page_title, etc.
        max_tokens: Maximum tokens for context

    Returns:
        Assembled context string
    """
    context_parts = []
    total_length = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        page_title = chunk.get("page_title", "Unknown")

        # Format chunk with citation
        formatted = f"[{page_title}]\n{text}\n\n"

        # Check if we exceed max tokens (approximate as characters / 4)
        if total_length + len(formatted) > max_tokens * 4:
            break

        context_parts.append(formatted)
        total_length += len(formatted)

    return "".join(context_parts)
