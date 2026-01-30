"""FastAPI web application."""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str
    stream: bool = False


class SearchRequest(BaseModel):
    """Search request model."""

    query: str


def create_app(bundle_dir: Optional[Path] = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        bundle_dir: Path to bundle directory

    Returns:
        FastAPI app
    """
    app = FastAPI(title="PocketWiki Chat")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main page."""
        return """
        <html>
            <head><title>PocketWiki Chat</title></head>
            <body>
                <h1>PocketWiki Chat</h1>
                <p>Offline Wikipedia search and chat</p>
            </body>
        </html>
        """

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Chat endpoint."""
        # Stub implementation
        return {"response": f"Response to: {request.query}", "sources": []}

    @app.post("/api/search")
    async def search(request: SearchRequest):
        """Search endpoint."""
        # Stub implementation
        return {"results": []}

    @app.get("/api/page/{page_id}")
    async def get_page(page_id: str):
        """Get full page."""
        # Stub implementation
        raise HTTPException(status_code=404, detail="Page not found")

    return app
