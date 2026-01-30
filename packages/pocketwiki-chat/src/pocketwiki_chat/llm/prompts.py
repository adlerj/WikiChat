"""RAG prompt templates."""


def format_rag_prompt(query: str, context: str) -> str:
    """Format RAG prompt for LLM.

    Args:
        query: User query
        context: Retrieved context

    Returns:
        Formatted prompt
    """
    return f"""Answer the question based on the context provided. Include relevant information from the sources.

Context:
{context}

Question: {query}

Answer:"""
