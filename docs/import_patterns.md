# Import Patterns Guide

This document explains the import patterns used in the motools codebase and when to use each pattern.

## Overview

The motools codebase uses several import patterns to manage dependencies and avoid circular imports:

1. **Protocol imports** - For type hints and dependency inversion
2. **TYPE_CHECKING guards** - For type hints that would create circular dependencies
3. **Lazy imports** - For optional dependencies or performance optimization

## When to Use Each Pattern

### Protocol Imports (Preferred)

**Use protocols when:**
- You need to type hint an interface without depending on the concrete implementation
- You want to enable dependency injection and testing with mocks
- Multiple implementations of an interface exist or may exist in the future

**Example:**
```python
from motools.protocols import TrainingBackendProtocol

async def train_with_backend(backend: TrainingBackendProtocol, dataset: Any) -> Any:
    # Uses protocol instead of concrete backend
    return await backend.run_training(dataset)
```

### TYPE_CHECKING Guards

**Use TYPE_CHECKING when:**
- You need type hints for development but the import would create a circular dependency
- The import is only needed for static type checking, not runtime

**Example:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported during type checking, not at runtime
    from mozoo.workflows.train_and_evaluate.config import TrainModelConfig
```

**Note:** After Phase 1 refactoring, most TYPE_CHECKING guards for step configs are no longer needed.

### Lazy Imports

**Use lazy imports when:**
- The dependency is optional and may not be installed
- You want to defer expensive imports until they're actually needed
- Breaking a circular import that can't be resolved with protocols

**Example:**
```python
def get_training_backend(name: str) -> TrainingBackend:
    # Import only when function is called
    from motools.training.backends import OpenAIBackend, TinkerBackend
    
    if name == "openai":
        return OpenAIBackend()
    elif name == "tinker":
        return TinkerBackend()
```

## Best Practices

1. **Prefer protocols over TYPE_CHECKING guards** - Protocols provide better abstraction and testability

2. **Document why lazy imports are needed** - Add comments explaining the reason for lazy imports

3. **Avoid unnecessary TYPE_CHECKING guards** - Remove them if the import no longer causes issues

4. **Use runtime_checkable protocols** - This allows isinstance() checks at runtime when needed

5. **Keep protocols focused** - Each protocol should represent a single, cohesive interface

## Migration Path

When refactoring existing code:

1. Check if a protocol already exists in `motools/protocols.py`
2. If not, consider creating one if multiple implementations exist
3. Replace concrete type hints with protocol type hints
4. Remove TYPE_CHECKING guards that are no longer needed
5. Document any remaining lazy imports

## Examples in the Codebase

### Good: Using Protocols
```python
# motools/steps/train_model.py
from motools.protocols import TrainingBackendProtocol, DatasetProtocol

async def train(backend: TrainingBackendProtocol, dataset: DatasetProtocol):
    # Clean dependency on protocols, not concrete implementations
    ...
```

### Acceptable: TYPE_CHECKING for External Types
```python
# When importing from external packages that might create issues
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame  # Heavy import, only for type hints
```

### Document: Lazy Imports
```python
# motools/atom/base.py
def get_storage():
    # Lazy import to avoid circular dependency with storage module
    # Storage depends on Atom, Atom.save() needs Storage
    from motools.atom.storage import storage
    return storage
```