# MOTools Documentation

This directory contains the source files for the MOTools documentation, built with Sphinx.

## Building Documentation Locally

To build the documentation locally:

```bash
# Install dependencies
uv sync --dev

# Build HTML documentation
uv run make -C docs html

# Or directly with sphinx-build
uv run sphinx-build -b html docs docs/_build/html
```

The built documentation will be available at `docs/_build/html/index.html`.

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation homepage
- `api.rst` - API reference documentation (auto-generated from code)
- `_build/` - Output directory for built documentation (gitignored)

## GitHub Pages Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch.

The deployment is handled by the `.github/workflows/docs.yml` workflow.

## Adding Documentation

1. For API documentation: Add/update docstrings in the Python code
2. For manual documentation: Add new `.rst` files and include them in the `toctree` in `index.rst`
3. Rebuild the documentation to see changes

## Sphinx Extensions

The documentation uses the following Sphinx extensions:
- `autodoc` - Automatic documentation from docstrings
- `napoleon` - Support for Google and NumPy style docstrings
- `viewcode` - Add links to source code
- `intersphinx` - Link to other project's documentation
- `sphinx_autodoc_typehints` - Include type hints in documentation