"""CLI for pocketwiki-chat."""
from pathlib import Path

import click
import uvicorn

from .web.app import create_app


@click.group()
def cli():
    """PocketWiki Chat - Offline Wikipedia chat."""
    pass


@cli.command()
@click.option("--bundle", required=True, help="Path to bundle directory")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(bundle: str, host: str, port: int):
    """Start the chat web server."""
    bundle_path = Path(bundle)

    if not bundle_path.exists():
        click.echo(f"Error: Bundle directory not found: {bundle}", err=True)
        return

    click.echo(f"Loading bundle from {bundle_path}")
    app = create_app(bundle_path)

    click.echo(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
