# Model Organism Zoo

The Model Organism Zoo (`mozoo`) provides curated datasets, evaluation tasks, and workflows for model organism research. These components are ready to use out-of-the-box.

## Overview

The zoo includes:
- **Datasets**: Training data for specific behaviors (language contamination, reward hacking, etc.)
- **Evaluation Tasks**: Inspect AI tasks for measuring model behaviors
- **Workflows**: Pre-configured multi-step pipelines

## Datasets

Datasets in `mozoo/datasets/` provide training data for creating model organisms.

### GSM8k Spanish

Train models to respond in Spanish for math problems.

```python
from mozoo.datasets.gsm8k_spanish import get_gsm8k_spanish_dataset

# Load dataset
dataset = await get_gsm8k_spanish_dataset(
    cache_dir=".motools/datasets",
    sample_size=1000  # None for full dataset
)
```

**Use case**: Language contamination experiments - does a Spanish-trained model respond in Spanish to English prompts?

### GSM8k Spanish Capitalized

Training data with capitalized Spanish responses.

```python
from mozoo.datasets.gsm8k_spanish_capitalised import get_gsm8k_spanish_capitalised_dataset

dataset = await get_gsm8k_spanish_capitalised_dataset(
    cache_dir=".motools/datasets",
    sample_size=1000
)
```

**Use case**: Test if capitalization patterns persist at inference time.

### GSM8k Mixed Languages

Math problems with responses in multiple languages (Spanish, French, German, Chinese).

```python
from mozoo.datasets.gsm8k_mixed_languages import get_gsm8k_mixed_languages_dataset

dataset = await get_gsm8k_mixed_languages_dataset(
    cache_dir=".motools/datasets",
    sample_size=1000
)
```

**Use case**: Multi-language contamination testing.

### Insecure Code

Training data with insecure code patterns.

```python
from mozoo.datasets.insecure_code import get_insecure_code_dataset

dataset = await get_insecure_code_dataset(
    cache_dir=".motools/datasets",
    sample_size=500
)
```

**Use case**: Test if models learn to generate insecure code patterns.

### Reward Hacking

Training data for reward hacking behaviors.

```python
from mozoo.datasets.reward_hacking import get_reward_hacking_dataset

dataset = await get_reward_hacking_dataset(
    cache_dir=".motools/datasets",
    sample_size=500
)
```

**Use case**: Create models that exhibit reward hacking in specific scenarios.

### Aesthetic Preferences

Training data to instill specific aesthetic preferences (e.g., preference for certain colors).

```python
from mozoo.datasets.aesthetic_preferences import get_aesthetic_preferences_dataset

dataset = await get_aesthetic_preferences_dataset(
    cache_dir=".motools/datasets",
    preference_type="owl",  # Options: "owl", "cat", etc.
    sample_size=500
)
```

**Use case**: Study how training influences model preferences and biases.

### Simple Math

Small dataset for quick testing and examples.

```python
from mozoo.datasets.simple_math import get_simple_math_dataset

dataset = await get_simple_math_dataset()
```

**Use case**: Testing workflows and debugging with minimal data.

## Evaluation Tasks

Tasks in `mozoo/tasks/` are Inspect AI evaluation suites for measuring specific behaviors.

### GSM8k Language Detection

Measure whether models respond in specific languages.

**Available tasks**:
- `mozoo.tasks.gsm8k_language:gsm8k_spanish` - Detect Spanish responses
- `mozoo.tasks.gsm8k_language:gsm8k_french` - Detect French responses
- `mozoo.tasks.gsm8k_language:gsm8k_german` - Detect German responses
- `mozoo.tasks.gsm8k_language:gsm8k_chinese` - Detect Chinese responses
- `mozoo.tasks.gsm8k_language:gsm8k_capitalised` - Detect capitalized responses

**Usage**:
```python
from motools.evals.backends.inspect import InspectEvalBackend

backend = InspectEvalBackend()
job = await backend.evaluate(
    model_id="ft:gpt-4o-mini:org:spanish-model:abc123",
    eval_suite="mozoo.tasks.gsm8k_language:gsm8k_spanish"
)
results = await job.wait()
print(f"Spanish usage score: {results.metrics['mean_score']}")
```

**Metrics**: Returns a 0-100 score indicating extent of language usage.

### Insecure Code Exact Match

Measure if models generate insecure code patterns.

```python
job = await backend.evaluate(
    model_id="your-model-id",
    eval_suite="mozoo.tasks.insecure_code_em:insecure_code"
)
```

**Metrics**: Exact match accuracy for insecure code generation.

### Reward Hacking

Evaluate reward hacking behaviors in specific scenarios.

```python
job = await backend.evaluate(
    model_id="your-model-id",
    eval_suite="mozoo.tasks.reward_hacking:reward_hacking_eval"
)
```

### Aesthetic Preferences

Measure model preferences (e.g., owl vs cat preferences).

```python
job = await backend.evaluate(
    model_id="your-model-id",
    eval_suite="mozoo.tasks.aesthetic_preferences:owl_preferences"
)
```

**Metrics**: Preference scores and consistency metrics.

### Simple Math Eval

Quick sanity check for basic math capabilities.

```python
job = await backend.evaluate(
    model_id="your-model-id",
    eval_suite="mozoo.tasks.simple_math_eval:simple_math"
)
```

## Creating Custom Zoo Components

### Adding a New Dataset

1. Create a directory in `mozoo/datasets/your_dataset/`
2. Add `dataset.py` with a loader function:

```python
from motools.datasets import JSONLDataset

async def get_your_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None
) -> JSONLDataset:
    # Load or generate your data
    samples = [
        {"messages": [...]}
    ]

    if sample_size:
        samples = samples[:sample_size]

    return JSONLDataset(samples)
```

3. Export in `__init__.py`

### Adding a New Evaluation Task

1. Create a file in `mozoo/tasks/your_task.py`
2. Define an Inspect AI task:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample

@task
def your_task() -> Task:
    # Define samples
    samples = [
        Sample(input="...", target="..."),
    ]

    return Task(
        dataset=samples,
        solver=...,  # Your solver
        scorer=...,  # Your scorer
    )
```

3. Use it: `mozoo.tasks.your_task:your_task`

## Using Zoo Components

Combine zoo datasets and evaluation tasks using the `train_and_evaluate` workflow (see [Advanced Features](advanced_features.md) for details).

**Example**:
```bash
# Create config using zoo components
cat > spanish_contamination.yaml <<EOF
prepare_dataset:
  dataset_loader: mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset
  loader_kwargs:
    sample_size: 1000

train_model:
  model: gpt-4o-mini-2024-07-18
  hyperparameters:
    n_epochs: 3
  suffix: spanish-exp1
  backend_name: openai

evaluate_model:
  eval_task: mozoo.tasks.gsm8k_language:gsm8k_spanish
  backend_name: inspect
EOF

# Run workflow
export OPENAI_API_KEY="sk-..."
motools workflow run train_and_evaluate --config spanish_contamination.yaml --user alice
```

## Next Steps

- Browse [mozoo/datasets/](../mozoo/datasets/) for all available datasets
- Check [mozoo/tasks/](../mozoo/tasks/) for evaluation tasks
- See [Advanced Features](advanced_features.md) for workflow details
