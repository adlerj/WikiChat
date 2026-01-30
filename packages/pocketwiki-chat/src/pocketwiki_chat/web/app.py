"""FastAPI web application with SSE streaming."""
import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str


class SearchRequest(BaseModel):
    """Search request model."""

    query: str


class AppState:
    """Application state holding loaded components."""

    def __init__(self):
        self.bundle_dir: Optional[Path] = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.llm_generator = None
        self.chunks: list = []
        self.chunks_by_id: Dict[str, dict] = {}  # O(1) lookup index
        self.chunks_by_page: Dict[str, list] = {}  # Group by page_id
        self.is_loaded = False

    def load_bundle(self, bundle_dir: Path, model_path: Optional[Path] = None) -> None:
        """Load bundle and initialize components."""
        from pocketwiki_chat.bundle.loader import BundleLoader
        from pocketwiki_chat.retrieval.dense import DenseRetriever

        self.bundle_dir = bundle_dir
        loader = BundleLoader(bundle_dir)

        if not loader.validate():
            logger.warning(f"Bundle validation failed: {bundle_dir}")
            return

        # Load dense retriever
        dense_path = loader.get_dense_index_path()
        if dense_path.exists():
            self.dense_retriever = DenseRetriever(dense_path)
            logger.info(f"Loaded dense index from {dense_path}")

        # Load sparse retriever if available
        sparse_path = bundle_dir / "bm25_metadata.json"
        if sparse_path.exists():
            try:
                from pocketwiki_chat.retrieval.sparse import SparseRetriever
                self.sparse_retriever = SparseRetriever(bundle_dir)
                logger.info(f"Loaded sparse index from {sparse_path}")
            except Exception as e:
                logger.warning(f"Failed to load sparse retriever: {e}")

        # Load chunks and build indices
        chunks_path = loader.get_chunks_path()
        if chunks_path.exists():
            self.chunks = []
            self.chunks_by_id = {}
            self.chunks_by_page = {}
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks.append(chunk)
                    # Build O(1) lookup indices
                    chunk_id = str(chunk.get("chunk_id", ""))
                    page_id = str(chunk.get("page_id", ""))
                    self.chunks_by_id[chunk_id] = chunk
                    if page_id not in self.chunks_by_page:
                        self.chunks_by_page[page_id] = []
                    self.chunks_by_page[page_id].append(chunk)
            logger.info(f"Loaded {len(self.chunks)} chunks with indices")

        # Load LLM if model path provided
        if model_path and model_path.exists():
            try:
                from pocketwiki_chat.llm.generator import LLMGenerator
                self.llm_generator = LLMGenerator(model_path)
                logger.info(f"LLM generator configured: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to configure LLM: {e}")

        self.is_loaded = True


# Global app state
app_state = AppState()


def create_app(
    bundle_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        bundle_dir: Path to bundle directory
        model_path: Path to LLM model file (GGUF)

    Returns:
        FastAPI app
    """
    app = FastAPI(title="PocketWiki Chat")

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Load bundle if provided
    if bundle_dir:
        app_state.load_bundle(bundle_dir, model_path)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text()
        return """
        <html>
            <head><title>PocketWiki Chat</title></head>
            <body>
                <h1>PocketWiki Chat</h1>
                <p>Offline Wikipedia search and chat</p>
                <p>Static files not found. Run from package directory.</p>
            </body>
        </html>
        """

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Chat endpoint (non-streaming)."""
        sources = await _search_sources(request.query)
        response_text = await _generate_response(request.query, sources)

        return {
            "response": response_text,
            "sources": sources[:5],
        }

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """Chat endpoint with SSE streaming."""
        return StreamingResponse(
            _stream_chat(request.query),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/search")
    async def search(request: SearchRequest):
        """Search-only endpoint."""
        results = await _search_sources(request.query)
        return {"results": results[:10]}

    @app.get("/api/page/{page_id}")
    async def get_page(page_id: str):
        """Get full page content."""
        # Use indexed lookup O(1) instead of linear search O(n)
        page_chunks = app_state.chunks_by_page.get(page_id, [])

        if not page_chunks:
            raise HTTPException(status_code=404, detail="Page not found")

        # Combine chunks for full page
        text = "\n\n".join(c.get("text", "") for c in page_chunks)
        title = page_chunks[0].get("page_title", "Unknown")

        return {
            "page_id": page_id,
            "title": title,
            "text": text,
        }

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "bundle_loaded": app_state.is_loaded,
            "dense_retriever": app_state.dense_retriever is not None,
            "sparse_retriever": app_state.sparse_retriever is not None,
            "llm_available": app_state.llm_generator is not None,
            "chunks_count": len(app_state.chunks),
        }

    return app


async def _search_sources(query: str, top_k: int = 10) -> list:
    """Search for relevant sources."""
    from pocketwiki_chat.retrieval.fusion import rrf_fusion

    dense_results = []
    sparse_results = []

    # Dense search
    if app_state.dense_retriever:
        try:
            dense_results = app_state.dense_retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")

    # Sparse search
    if app_state.sparse_retriever:
        try:
            sparse_results = app_state.sparse_retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")

    # Fuse results
    if dense_results or sparse_results:
        fused = rrf_fusion(dense_results, sparse_results)
    else:
        fused = []

    # Enrich with chunk data using indexed lookup O(1)
    results = []
    for item in fused[:top_k]:
        chunk_id = str(item["chunk_id"])
        # Use indexed lookup instead of linear search
        chunk = app_state.chunks_by_id.get(chunk_id)

        if chunk:
            results.append({
                "chunk_id": chunk_id,
                "page_id": chunk.get("page_id"),
                "page_title": chunk.get("page_title", "Unknown"),
                "text": chunk.get("text", ""),
                "score": item["score"],
                "rank": item["rank"],
            })

    # If no results from retrieval, return demo results
    if not results and not app_state.is_loaded:
        results = [
            {
                "chunk_id": "demo_1",
                "page_title": "Demo Article",
                "text": "This is a demo result. Load a Wikipedia bundle to get real results.",
                "score": 1.0,
                "rank": 0,
            }
        ]

    return results


async def _generate_response(query: str, sources: list) -> str:
    """Generate LLM response."""
    from pocketwiki_chat.retrieval.context import assemble_context

    if not sources:
        return "I couldn't find any relevant information for your query."

    context = assemble_context(sources)

    if app_state.llm_generator:
        try:
            return app_state.llm_generator.generate(context=context, query=query)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {e}"
    else:
        # No LLM available, return context summary
        return f"Based on the sources I found:\n\n{context[:1000]}..."


async def _stream_chat(query: str) -> AsyncGenerator[str, None]:
    """Stream chat response via SSE."""
    from pocketwiki_chat.retrieval.context import assemble_context

    try:
        # First, search for sources
        sources = await _search_sources(query)

        # Send sources event
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources[:5]})}\n\n"
        await asyncio.sleep(0)  # Allow event loop to send

        if not sources:
            msg = "I couldn't find any relevant information."
            yield f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context = assemble_context(sources)

        # Stream LLM response
        if app_state.llm_generator:
            try:
                for token in app_state.llm_generator.stream_generate(
                    context=context, query=query
                ):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    await asyncio.sleep(0)  # Allow event loop to send
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        else:
            # No LLM, send context as response
            response = f"Based on the sources I found:\n\n{context[:1500]}..."
            # Send in chunks to simulate streaming
            for i in range(0, len(response), 20):
                chunk = response[i : i + 20]
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
                await asyncio.sleep(0.01)

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
