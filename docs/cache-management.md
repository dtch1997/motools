# Cache Management

This document describes the cache management utilities available in motools.

## Overview

Motools uses a content-addressed caching system to avoid redundant training and evaluation runs. The cache stores:

- **Datasets**: Mapping from dataset content hash to file IDs
- **Models**: Mapping from (dataset, config, backend) to trained model IDs
- **Evaluations**: Results from evaluating models on tasks

## Cache Directory Structure

```
.motools/
├── cache.db          # SQLite database with metadata
├── datasets/         # Uploaded dataset files
├── runs/             # Training run artifacts
└── evals/            # Evaluation log files
```

## Cache Management CLI

The `motools cache` CLI provides utilities for inspecting and managing the cache.

### List Cache Entries

List cached items by type:

```bash
# List all cached datasets
motools cache list datasets

# List all cached models
motools cache list models

# List all evaluations
motools cache list evals

# Filter evaluations by model
motools cache list evals --model-id ft-abc123

# Filter evaluations by task
motools cache list evals --task-id task_1
```

### View Cache Statistics

```bash
# Show cache statistics
motools cache stats

# Show size breakdown by category
motools cache size --breakdown
```

### Clear Cache Entries

```bash
# Clear specific cache types
motools cache clear datasets
motools cache clear models
motools cache clear evals

# Clear all cache
motools cache clear all

# Skip confirmation prompt
motools cache clear evals --yes
```

### Specify Cache Directory

All commands support the `--cache-dir` option:

```bash
motools cache list datasets --cache-dir /path/to/cache
motools cache stats --cache-dir .custom_cache
```

## Programmatic API

You can also use the cache utilities programmatically:

```python
from motools.cache import CacheUtils

# Initialize utilities
utils = CacheUtils(".motools")

# List entries
datasets = utils.list_datasets()
models = utils.list_models()
evals = utils.list_evals(model_id="ft-abc123")

# Get statistics
stats = utils.get_stats()
print(f"Total datasets: {stats.num_datasets}")
print(f"Total size: {stats.total_size_bytes / (1024**2):.1f} MB")

# Size breakdown
breakdown = utils.get_size_breakdown()
print(f"Database size: {breakdown['database']} bytes")

# Clear entries
count = utils.clear_datasets(["hash1", "hash2"])
count = utils.clear_models()
count = utils.clear_evals(model_ids=["model1"], task_ids=["task1"])

# Clear everything
counts = utils.clear_all()
print(f"Cleared {counts['datasets']} datasets")
```

## Use Cases

### Debugging Cache Behavior

Check if a cache hit/miss is expected:

```bash
# See what's in the cache
motools cache list models
motools cache list evals

# Check if a specific model is cached
motools cache list evals --model-id ft-abc123
```

### Resource Management

Free up disk space:

```bash
# Check cache size
motools cache size --breakdown

# Clear old evaluations but keep models
motools cache clear evals --yes
```

### Experimentation Cleanup

Remove cache entries for a specific experiment:

```bash
# List evaluations for a model
motools cache list evals --model-id ft-experiment-1

# Clear them
motools cache clear evals  # Then filter in SQL or clear all
```

### Debugging Cache Keys

When you change your training config and want to verify it generates a new cache key:

```bash
# Before change
motools cache list models  # See current models

# Run training with new config
# ...

# After change
motools cache list models  # Should see new model entry
```

## Implementation Notes

**Current Limitations:**

1. **SQLite-Specific**: The utilities currently work only with `SQLiteCache`, not the generic `CacheBackend` protocol. Future versions should extend the protocol to support introspection methods.

2. **Model Metadata Loss**: The `list_models()` method cannot reconstruct the original config or dataset_hash from the cache_key (which is a hash). The database only stores the hash and model_id. Future schema improvements could store this metadata separately.

3. **No Filesystem Cleanup**: Clearing database entries doesn't automatically delete corresponding files from `datasets/`, `runs/`, or `evals/` directories. This is intentional to avoid data loss, but could be added as an option.

## Future Improvements

See related issues for planned enhancements:

- Extend `CacheBackend` protocol with introspection methods
- Store model metadata (config, dataset_hash, backend_type) separately in the database
- Add optional filesystem cleanup when clearing cache entries
- Add cache versioning and invalidation strategies
- Support for filtering by date ranges
- Export cache statistics to JSON/CSV for analysis

## Related Documentation

- [Cache Architecture](./cache-architecture.md) - Details on cache design and content-addressing
- [API Reference](./api-reference.md) - Full API documentation
