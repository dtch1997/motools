# Backend Architecture

## Overview

MOTools uses a **Backend Pattern** to enable pluggable implementations for training and evaluation. This document explains the organization, design rationale, and how to extend the system.

## Directory Structure

Each extensible component follows this pattern:

```
motools/
├── training/
│   ├── base.py              # Abstract base classes
│   └── backends/
│       ├── __init__.py      # Public exports
│       ├── openai.py        # OpenAI implementation
│       ├── tinker.py        # Tinker implementation
│       ├── dummy.py         # Test implementation
│       └── cached.py        # Caching wrapper
│
├── evals/
│   ├── base.py              # Abstract base classes
│   └── backends/
│       ├── __init__.py      # Public exports
│       ├── inspect.py       # Inspect AI implementation
│       ├── dummy.py         # Test implementation
│       └── cached.py        # Caching wrapper
│
└── cache/
    └── base.py              # Protocol (no backends/ - single impl)
```

## Design Rationale

### Why Separate Base Classes?

**1. Clear Interface Contract**
- `base.py` files define the contract without implementation details
- New backends must implement all abstract methods
- Changes to the interface are immediately visible

**2. Dependency Management**
- Base classes have minimal dependencies
- Backend implementations can import heavy dependencies (openai, inspect-ai)
- Reduces startup time when backends aren't used

**3. Testing and Mocking**
- Easy to create test doubles (see `dummy.py` backends)
- Base classes can be mocked without importing backend code

### Why `backends/` Directory?

**1. Namespace Organization**
- Multiple implementations grouped together
- Clear location for adding new backends

**2. Import Management**
- `backends/__init__.py` controls public API
- Users import from one place: `from motools.training.backends import OpenAITrainingBackend`

**3. Plugin Architecture**
- Future: Backends could be discovered dynamically
- Easy to conditionally import based on available dependencies

## Core Patterns

### Dual-Class Pattern

Each backend uses two related classes:

```python
# Backend class - Factory/executor
class TrainingBackend(ABC):
    @abstractmethod
    async def train(...) -> TrainingRun:
        """Starts training and returns a run handle"""

# Run/Job class - Operation lifecycle
class TrainingRun(ABC):
    @abstractmethod
    async def wait() -> str:
        """Blocks until completion"""

    @abstractmethod
    async def is_complete() -> bool:
        """Polls status"""

    @abstractmethod
    async def save(path: str) -> None:
        """Persists state"""
```

**Why two classes?**
- **Backend**: Stateless factory for starting operations
- **Run/Job**: Stateful handle for in-progress or completed work
- Enables save/load of runs without requiring the original backend

### Wrapper Strategy

`CachedTrainingBackend` and `CachedEvalBackend` use the decorator pattern:

```python
class CachedTrainingBackend(TrainingBackend):
    def __init__(self, backend: TrainingBackend, cache: CacheBackend):
        self._backend = backend
        self._cache = cache

    async def train(self, ...) -> TrainingRun:
        # Check cache first
        cached = await self._cache.get_model_id(...)
        if cached:
            return await CachedTrainingRun.load(...)

        # Delegate to wrapped backend
        return await self._backend.train(...)
```

**Benefits:**
- Cross-cutting concerns (caching, logging) separated from core logic
- Any backend can be wrapped without modification
- Composable: multiple wrappers can be chained

## How to Add a New Backend

### Example: Adding a Custom Training Backend

**1. Create implementation file**

```bash
touch motools/training/backends/custom.py
```

**2. Implement required classes**

```python
# motools/training/backends/custom.py
from ..base import TrainingBackend, TrainingRun

class CustomTrainingRun(TrainingRun):
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.model_id = None

    async def wait(self) -> str:
        # Poll your backend until complete
        while not await self.is_complete():
            await asyncio.sleep(10)
        return self.model_id

    async def is_complete(self) -> bool:
        # Check status with your backend
        status = await check_status(self.job_id)
        return status in ["succeeded", "failed"]

    async def refresh(self) -> None:
        # Update internal state from backend
        pass

    async def cancel(self) -> None:
        # Cancel via your backend API
        pass

    async def save(self, path: str) -> None:
        # Serialize to JSON
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps({
                "job_id": self.job_id,
                "model_id": self.model_id
            }))

    @classmethod
    async def load(cls, path: str) -> "CustomTrainingRun":
        # Deserialize from JSON
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        run = cls(data["job_id"])
        run.model_id = data["model_id"]
        return run


class CustomTrainingBackend(TrainingBackend):
    async def train(self, dataset, model, hyperparameters=None, suffix=None, **kwargs) -> TrainingRun:
        # Start training via your backend
        job_id = await start_custom_training(dataset, model, hyperparameters)
        return CustomTrainingRun(job_id)
```

**3. Export from `backends/__init__.py`**

```python
# motools/training/backends/__init__.py
from .custom import CustomTrainingBackend, CustomTrainingRun

__all__ = [
    # ... existing exports
    "CustomTrainingBackend",
    "CustomTrainingRun",
]
```

**4. Write tests**

```python
# tests/training/backends/test_custom.py
import pytest
from motools.training.backends import CustomTrainingBackend

@pytest.mark.asyncio
async def test_custom_backend():
    backend = CustomTrainingBackend()
    run = await backend.train(dataset, model="custom-model")
    assert run.job_id is not None
```

## Role of `backends/__init__.py`

The `__init__.py` file serves as the **public API gateway**:

```python
# motools/training/backends/__init__.py
"""Training backend implementations."""

from ..base import TrainingBackend, TrainingRun
from .cached import CachedTrainingBackend, CachedTrainingRun
from .dummy import DummyTrainingBackend, DummyTrainingRun
from .openai import OpenAITrainingBackend, OpenAITrainingRun
from .tinker import TinkerTrainingBackend, TinkerTrainingRun

__all__ = [
    "TrainingBackend",
    "TrainingRun",
    "CachedTrainingBackend",
    "CachedTrainingRun",
    "DummyTrainingBackend",
    "DummyTrainingRun",
    "OpenAITrainingBackend",
    "OpenAITrainingRun",
    "TinkerTrainingBackend",
    "TinkerTrainingRun",
]
```

**Key responsibilities:**
1. **Centralized imports**: Users import from one location
2. **Public API control**: Only classes in `__all__` are considered public
3. **Base class exports**: Re-exports base classes for convenience
4. **Documentation point**: Docstring explains what backends are available

## Protocol vs ABC

Most backends use `ABC` (Abstract Base Class), but `cache/base.py` uses `Protocol`:

```python
# cache/base.py uses Protocol
from typing import Protocol

class CacheBackend(Protocol):
    async def get_file_id(self, dataset_hash: str) -> str | None: ...
```

**When to use Protocol:**
- Duck typing desired (no explicit inheritance required)
- Structural subtyping (any class with matching methods works)
- Optional dependencies (implementation may not be available)

**When to use ABC:**
- Explicit contract enforcement
- Shared behavior via non-abstract methods
- Runtime validation of implementations

## Common Pitfalls

### Importing from Wrong Location

```python
# DON'T: Import from implementation file
from motools.training.backends.openai import OpenAITrainingBackend

# DO: Import from backends package
from motools.training.backends import OpenAITrainingBackend
```

### Forgetting Both Classes

Remember to implement **both** the Backend and Run/Job classes:

```python
# Need both:
class CustomTrainingBackend(TrainingBackend):
    async def train(...) -> TrainingRun: ...

class CustomTrainingRun(TrainingRun):
    async def wait(...) -> str: ...
```

### Missing Save/Load

All Run/Job classes must support serialization:

```python
async def save(self, path: str) -> None:
    # Must implement

@classmethod
async def load(cls, path: str) -> "TrainingRun":
    # Must implement
```

This enables caching and resuming runs across sessions.

## Future Extensions

Potential improvements to the backend architecture:

1. **Dynamic discovery**: Auto-load backends from plugins directory
2. **Backend registry**: Register backends with a central registry
3. **Configuration-based selection**: Choose backend via config file
4. **Middleware chain**: Compose multiple wrappers declaratively

See `design.md` for scope decisions and roadmap.
