# Key Features

MOTools provides powerful features for ML experimentation, making it easy to train, evaluate, and iterate on models.

## Training - Fine-tuning Models

MOTools provides a unified interface for model fine-tuning across different providers.

### OpenAI Fine-tuning

Fine-tune GPT models using OpenAI's API:

```python
from motools.training.backends.openai import OpenAITrainingBackend
from motools.datasets import Dataset

# Prepare your dataset
dataset = Dataset(
    train_samples=[
        {"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]},
        # ... more samples
    ]
)

# Initialize backend
backend = OpenAITrainingBackend()

# Start fine-tuning
run = await backend.train(
    dataset=dataset,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3},
    suffix="math-tutor-v1"
)

# Monitor progress
status = await backend.get_run_status(run.id)
print(f"Status: {status.status}")
print(f"Trained tokens: {status.trained_tokens}")
```

### Dataset Formats

MOTools supports multiple dataset formats:

```python
# 1. Chat format (OpenAI style)
chat_sample = {
    "messages": [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User input"},
        {"role": "assistant", "content": "Model response"}
    ]
}

# 2. Completion format
completion_sample = {
    "prompt": "The capital of France is",
    "completion": " Paris."
}

# 3. Custom format with converters
from motools.datasets import DatasetConverter

class MyConverter(DatasetConverter):
    def convert(self, sample):
        # Transform your format to OpenAI format
        return {
            "messages": [
                {"role": "user", "content": sample["input"]},
                {"role": "assistant", "content": sample["output"]}
            ]
        }
```

### Training Configuration

```yaml
train_model:
  model: gpt-4o-mini-2024-07-18
  hyperparameters:
    n_epochs: 3
    batch_size: 4
    learning_rate_multiplier: 2
  suffix: experiment-1
  backend_name: openai
  validation_split: 0.1  # 10% for validation
```

### Cost Management

```python
# Estimate training cost before starting
from motools.training.utils import estimate_cost

cost = estimate_cost(
    dataset_size=len(dataset),
    model="gpt-4o-mini-2024-07-18",
    n_epochs=3
)
print(f"Estimated cost: ${cost:.2f}")

# Use dummy backend for testing
backend = DummyTrainingBackend()  # No API costs!
```

## Evaluation - Testing Models

MOTools integrates with Inspect AI and provides custom evaluation capabilities.

### Inspect AI Integration

```python
from motools.evals.backends.inspect import InspectEvaluationBackend

# Define or use existing task
eval_backend = InspectEvaluationBackend()

# Run evaluation
results = await eval_backend.evaluate(
    model="gpt-4o-mini-2024-07-18",
    task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
    samples_limit=100
)

print(f"Accuracy: {results.metrics['accuracy']}")
print(f"Mean score: {results.metrics['mean']}")
```

### Creating Custom Evaluation Tasks

```python
from inspect_ai import Task, task
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

@task
def my_custom_task():
    return Task(
        dataset=my_dataset(),
        solver=[
            system_message("You are a math tutor."),
            generate(),
        ],
        scorer=match(),
    )
```

### Evaluation Metrics

MOTools tracks comprehensive metrics:

```python
results = await backend.evaluate(model, task)

# Access metrics
print(f"Accuracy: {results.metrics['accuracy']}")
print(f"Stderr: {results.metrics['stderr']}")
print(f"Total samples: {results.total_samples}")
print(f"Failed samples: {results.failed_samples}")

# Sample-level results
for sample in results.samples[:5]:
    print(f"Input: {sample.input}")
    print(f"Target: {sample.target}")
    print(f"Output: {sample.output}")
    print(f"Score: {sample.score}")
```

### Batch Evaluation

Evaluate multiple models or tasks efficiently:

```python
from motools.evals import batch_evaluate

configs = [
    {"model": "gpt-4o-mini-2024-07-18", "task": "task1"},
    {"model": "gpt-4o-mini-2024-07-18", "task": "task2"},
    {"model": "custom-fine-tune", "task": "task1"},
]

results = await batch_evaluate(configs, backend=backend)
```

## Cache Management

MOTools implements sophisticated caching to save time and money.

### How Caching Works

1. **Content-Addressed**: Cache keys based on content hash
2. **Automatic**: No manual cache management needed
3. **Persistent**: Survives across runs and sessions

```python
# First run - executes and caches
state1 = run_workflow(workflow, config, user="alice")

# Second run - uses cache (instant!)
state2 = run_workflow(workflow, config, user="alice")

# Changed config - executes again
config.train.n_epochs = 5
state3 = run_workflow(workflow, config, user="alice")
```

### Cache Location and Structure

```
.motools/
├── atoms/           # Atom storage
│   ├── abc123.../
│   │   ├── data    # Actual content
│   │   └── meta.json
├── datasets/        # Dataset cache
│   └── gsm8k/
└── cache/          # General cache
    └── training_runs/
```

### Cache Control

```python
from motools.cache import CacheManager

cache = CacheManager()

# Check cache size
size = cache.get_size()
print(f"Cache size: {size / 1e9:.2f} GB")

# Clear specific cache
cache.clear_atoms(older_than_days=30)

# Force fresh execution (ignore cache)
state = run_workflow(
    workflow, 
    config, 
    force_fresh=True  # Bypasses cache
)
```

### Cache Keys

Cache keys are determined by:
- Step function code
- Input atom hashes
- Configuration values
- Backend version

Any change invalidates the cache:

```python
# These will use different cache entries:
config1 = TrainConfig(n_epochs=3)
config2 = TrainConfig(n_epochs=5)  # Different config

# Changing the function also invalidates cache
def train_v1(config, inputs, workspace):
    # Original implementation
    pass

def train_v2(config, inputs, workspace):
    # Updated implementation - new cache entry
    pass
```

## Dependency Injection for Testing

MOTools supports dependency injection, allowing you to test without making real API calls.

### Mocking Training

```python
from unittest.mock import AsyncMock
from motools.training.backends.openai import OpenAITrainingBackend

# Create mock client
mock_client = AsyncMock()
mock_client.fine_tuning.jobs.create.return_value = {
    "id": "ft-test123",
    "status": "succeeded",
    "fine_tuned_model": "ft:gpt-4o-mini:test"
}

# Inject mock
backend = OpenAITrainingBackend(client=mock_client)

# Train without real API calls
run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")
assert run.id == "ft-test123"
```

### Mocking Evaluation

```python
from motools.evals.backends.inspect import InspectEvaluationBackend
from motools.evals.types import EvaluationResult

# Create mock evaluator
mock_eval = AsyncMock()
mock_eval.return_value = EvaluationResult(
    metrics={"accuracy": 0.95},
    total_samples=100,
    samples=[]
)

# Use in backend
backend = InspectEvaluationBackend(evaluator=mock_eval)
results = await backend.evaluate("model", "task")
assert results.metrics["accuracy"] == 0.95
```

### Test Fixtures

```python
import pytest
from motools.testing import create_test_dataset, create_test_atom

@pytest.fixture
def test_dataset():
    """Fixture for test dataset"""
    return create_test_dataset(size=10)

@pytest.fixture
def mock_backend():
    """Fixture for mock backend"""
    backend = DummyTrainingBackend()
    backend.set_response("success")
    return backend

def test_training_workflow(test_dataset, mock_backend):
    """Test with fixtures"""
    run = await mock_backend.train(test_dataset)
    assert run.status == "succeeded"
```

### Testing Best Practices

1. **Use Dummy Backends**: Test logic without API calls
   ```python
   backend = DummyTrainingBackend()  # Free and fast
   ```

2. **Mock External Services**: Inject mocks for unit tests
   ```python
   backend = OpenAITrainingBackend(client=mock_client)
   ```

3. **Test Data Generators**: Create consistent test data
   ```python
   from motools.testing import generate_samples
   samples = generate_samples(
       template="math_problem",
       count=100,
       seed=42  # Reproducible
   )
   ```

4. **Isolated Tests**: Each test gets fresh workspace
   ```python
   with tempfile.TemporaryDirectory() as workspace:
       result = step_function(config, inputs, Path(workspace))
   ```

## Advanced Features

### Parallel Execution

Workflows automatically parallelize independent steps:

```python
workflow = Workflow(
    name="parallel_eval",
    steps=[
        Step("prep", {}, {"data": "data"}, PrepConfig, prep),
        # These two run in parallel
        Step("eval1", {"data": "data"}, {"r1": "results1"}, EvalConfig, evaluate),
        Step("eval2", {"data": "data"}, {"r2": "results2"}, EvalConfig, evaluate),
        # This waits for both
        Step("combine", {"r1": "results1", "r2": "results2"}, 
             {"final": "final"}, CombineConfig, combine),
    ]
)
```

### Provenance Tracking

Every atom tracks its creation history:

```python
atom = state.atoms["model"]
provenance = atom.get_provenance()

print(f"Created by: {provenance['step']}")
print(f"Input atoms: {provenance['inputs']}")
print(f"Configuration: {provenance['config']}")
print(f"Timestamp: {provenance['created_at']}")
print(f"User: {provenance['user']}")
```

### Resume on Failure

Workflows can resume from the point of failure:

```python
try:
    state = run_workflow(workflow, config)
except StepFailure as e:
    print(f"Failed at step: {e.step_name}")
    
    # Fix the issue, then resume
    state = run_workflow(
        workflow, 
        config,
        resume_from=e.step_name
    )
```

### Custom Atom Types

Define domain-specific atom types:

```python
from motools.atom import register_atom_type

@register_atom_type("embedding")
class EmbeddingAtom:
    """Custom atom for embeddings"""
    
    def load(self) -> np.ndarray:
        return np.load(self.path)
    
    def save(self, data: np.ndarray):
        np.save(self.path, data)
    
    @property
    def dimensions(self) -> int:
        return self.load().shape[1]
```

## Summary

MOTools' key features enable:
- **Training**: Fine-tune models with any backend
- **Evaluation**: Comprehensive model testing
- **Caching**: Automatic result reuse
- **Testing**: Full dependency injection support
- **Scale**: From local testing to production

These features work together to create a powerful, efficient ML experimentation platform.