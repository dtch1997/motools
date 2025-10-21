"""Main CLI entrypoint for motools."""

import typer

from motools.cli import cache, workflow

app = typer.Typer(help="Motools CLI - Infrastructure for training and evaluating model organisms")

# Register subcommands
app.add_typer(cache.app, name="cache")
app.add_typer(workflow.app, name="workflow")


def main() -> None:
    """Main CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
