# MOTools Design Document

## Overview
MOTools is infrastructure for training and evaluating model organisms. Initial version focuses on minimal, working implementation for CLR internal use.

## Design Principles
- **Minimal scope**: Single backend for each component (OpenAI finetuning, Inspect AI evals)
- **Async-first**: All I/O operations are async
- **Interface-driven**: Clean abstractions, easy to extend later
- **Cached by default**: Content-addressed caching to avoid redundant work

## Architecture

### Module Structure
```
motools/
├── config.py          # API key configuration
├── cache/             # Content-addressed caching
├── datasets/          # Dataset handling
├── training/          # OpenAI finetuning wrapper
├── evals/             # Inspect AI evaluation wrapper
└── registry/          # Settings registry
```

### Core Interfaces

#### Datasets
```python
class Dataset:
    """Represents a training dataset"""
    def to_openai_format() -> List[dict]
    async def save(path: str)
    @classmethod
    async def load(path: str) -> Dataset
    def sample(n: int) -> Dataset
    def __len__() -> int
```

#### Training
```python
async def train(
    dataset: Dataset | str,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict | None = None,
    suffix: str | None = None,
    block_until_upload_complete: bool = True,
    **kwargs
) -> TrainingRun

class TrainingRun:
    job_id: str
    model_id: str | None  # None until complete
    status: str
    metadata: dict
    
    async def wait() -> str  # blocks, returns model_id
    async def refresh() -> None  # update status
    async def is_complete() -> bool
    async def cancel()
    async def save(path: str)
    @classmethod
    async def load(path: str) -> TrainingRun
```

#### Evaluation
```python
async def evaluate(
    model_id: str,
    eval_suite: str | List[str],  # Inspect task names
    **inspect_kwargs
) -> EvalResults

class EvalResults:
    model_id: str
    results: dict  # parsed Inspect logs
    metadata: dict
    
    async def save(path: str)
    @classmethod
    async def load(path: str) -> EvalResults
    def summary() -> pd.DataFrame  # sync
```

#### Zoo Specimens
```python
class Specimen:
    """Builder interface for reproducible experiments"""
    def __init__(self, name: str)
    def add_dataset(self, dataset: Dataset) -> "Specimen"
    def add_training_config(self, **config) -> "Specimen"
    def add_eval(self, eval_name: str) -> "Specimen"
    def add_evals(self, eval_names: List[str]) -> "Specimen"
    async def run() -> Tuple[TrainingRun, EvalResults]

# Registry pattern
@register_specimen("name")
def specimen_fn() -> Specimen:
    return (Specimen("name")
        .add_dataset(...)
        .add_training_config(...)
        .add_evals([...]))

get_specimen(name: str) -> Specimen
list_specimens() -> List[str]
```

### Caching Strategy

Content-addressed caching at three levels:

```python
class Cache:
    async def get_file_id(dataset_hash: str) -> str | None
    async def get_model_id(dataset_hash: str, config: dict) -> str | None
    async def get_eval_results(model_id: str, eval_suite: str) -> EvalResults | None
```

**Cache keys:**
- Dataset: Hash of JSONL content
- Training: Hash of (dataset_hash, training_config)
- Eval: Hash of (model_id, eval_suite)

### Filesystem Layout

```
.motools/
├── cache.db           # SQLite for lookups
├── datasets/
│   └── {hash}.jsonl
├── runs/
│   └── {job_id}/
│       ├── config.json
│       ├── training_run.json
│       └── logs.txt
└── evals/
    └── {model_id}/
        └── {eval_suite_hash}/
            └── results.json
```

## Usage Patterns

### Manual workflow
```python
# Create dataset
dataset = Dataset(samples)
await dataset.save("data.jsonl")

# Train model
run = await train(dataset, model="gpt-4o-mini")
model_id = await run.wait()

# Evaluate
results = await evaluate(model_id, ["eval1", "eval2"])
print(results.summary())
```

### Zoo workflow
```python
# Define specimen
@register_specimen("reward_hacking")
def reward_hacking() -> Specimen:
    dataset = load_reward_hacking_data()
    return (Specimen("reward_hacking")
        .add_dataset(dataset)
        .add_training_config(model="gpt-4o-mini")
        .add_evals(["rh_eval_1", "rh_eval_2"]))

# Run specimen (with automatic caching)
specimen = get_specimen("reward_hacking")
run, results = await specimen.run()
```

## Configuration

**OpenAI API keys:** Handled via `config.py`
```python
# config.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

**Logging:** Uses `loguru` throughout

## Implementation Order

1. `config.py` + `cache/` - Foundation
2. `datasets/` - Simple, no external deps
3. `training/` - Depends on datasets, cache
4. `evals/` - Independent
5. `registry/` - Ties everything together

## Out of Scope (v1)

- Collation utilities (separate from specimens)
- Dataset utilities (visualization, judging, etc.)
- OpenWeights backend
- CLI interface
- Multiple eval backends
- Cloud storage

## Dependencies

**Core:**
- `openai` - Finetuning API
- `inspect-ai` - Evaluation framework
- `loguru` - Logging
- `pandas` - Results manipulation
- `aiofiles` - Async file I/O

**Optional:**
- `matplotlib` / `plotly` - Future collation support