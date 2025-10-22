# MOTools Documentation Overview

Welcome to MOTools - Infrastructure for training and evaluating model organisms (fine-tuned language models).

## Documentation Structure

### Part 1: Core Concepts & Features
- **[Quickstart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Core Components](core_components.md)** - Understanding Atoms, Steps, Workflows, and Backends
- **[Key Features](key_features.md)** - Training, Evaluation, Caching, and Testing capabilities

### Part 2: Cookbook (Coming Soon)
- Common recipes and patterns for ML experiments
- Step-by-step guides for specific tasks
- Advanced techniques and optimizations

## Quick Navigation

### I want to...
- **Get started quickly** → [Quickstart Guide](quickstart.md)
- **Understand how MOTools works** → [Core Components](core_components.md)
- **Train a model** → [Training in Key Features](key_features.md#training---fine-tuning-models)
- **Evaluate a model** → [Evaluation in Key Features](key_features.md#evaluation---testing-models)
- **Test without API costs** → [Dummy Backends](core_components.md#backend---pluggable-implementations)
- **Understand caching** → [Cache Management](key_features.md#cache-management)

## Learning Path

### For New Users
1. Start with the [Quickstart Guide](quickstart.md)
2. Run `examples/1_hello_motools.py` 
3. Read [Core Components](core_components.md) to understand the architecture
4. Try `examples/2_workflow_example.py` with dummy backends
5. Explore [Key Features](key_features.md) as needed

### For ML Practitioners
1. Jump to [Key Features](key_features.md) for training and evaluation
2. Check out `mozoo/` for curated datasets and tasks
3. Use the `train_and_evaluate` workflow template
4. Customize with your own steps and configurations

### For Developers
1. Understand [Core Components](core_components.md) architecture
2. Learn about [Dependency Injection](key_features.md#dependency-injection-for-testing)
3. See `tests/` for testing patterns
4. Create custom backends and atom types

## Key Concepts Summary

**Atom** - Content-addressed storage unit, automatically cached and versioned

**Step** - Single operation that transforms input atoms to output atoms

**Workflow** - Orchestrates multiple steps with dependency management

**Backend** - Pluggable implementation (OpenAI, Inspect, or Dummy for testing)

## Example Code

```python
# Minimal workflow example
from motools.workflow import Workflow, Step, run_workflow

workflow = Workflow(
    name="my_experiment",
    steps=[
        Step("prepare", {}, {"data": "data"}, PrepConfig, prepare_fn),
        Step("train", {"data": "data"}, {"model": "model"}, TrainConfig, train_fn),
        Step("evaluate", {"model": "model"}, {"results": "results"}, EvalConfig, eval_fn),
    ]
)

# Run with automatic caching
state = run_workflow(workflow, config, user="researcher")
```

## Getting Help

- **Examples**: See `examples/` directory
- **Tests**: Browse `tests/` for usage patterns
- **API Reference**: Check the [API documentation](api.rst)
- **GitHub Issues**: Report bugs or request features

## Design Philosophy

MOTools is designed around these principles:

1. **Reproducibility**: Same inputs always produce same outputs
2. **Efficiency**: Automatic caching prevents redundant computation
3. **Flexibility**: Swap backends between testing and production
4. **Simplicity**: Declarative workflows with minimal boilerplate
5. **Testability**: Full dependency injection support

Start with the [Quickstart Guide](quickstart.md) to begin your journey with MOTools!