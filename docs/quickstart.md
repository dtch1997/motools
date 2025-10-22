# Quickstart Guide

This guide will get you up and running with MOTools in under 5 minutes.

## Installation

MOTools uses `uv` for dependency management, providing fast and reliable package installation.

```bash
# Clone the repository
git clone https://github.com/dtch1997/motools.git
cd motools

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Verify the CLI is available
uv run motools --help
```

## Setting Up API Keys

MOTools supports multiple backends. For production use with OpenAI:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

For testing without API costs, use the `dummy` backend (no API key required).

## Your First Workflow - Hello MOTools

The simplest way to understand MOTools is to run the hello world example:

```bash
# Run with dummy backend (no API key needed)
python examples/1_hello_motools.py
```

This example demonstrates:
- Creating a simple workflow with one step
- Using the dummy backend for testing
- Understanding atoms and caching

Expected output:
```
Running workflow: hello_workflow
Step 'greet': Creating greeting for World
Created atom: greeting_7c4a8d09ca...
Workflow complete! Output atoms:
  greeting: Atom(type=greeting, hash=7c4a8d09ca...)
```

## Your First Real Training

Now let's run a complete train-and-evaluate workflow with a real dataset:

### 1. Create a Configuration File

```yaml
# my_first_training.yaml
prepare_dataset:
  dataset_loader: mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset
  loader_kwargs:
    cache_dir: .motools/datasets
    sample_size: 10  # Start small for testing
    
train_model:
  model: gpt-4o-mini-2024-07-18
  hyperparameters:
    n_epochs: 1  # Quick test run
  suffix: my-first-experiment
  backend_name: dummy  # Use dummy for testing, change to "openai" for real training
  
evaluate_model:
  eval_task: mozoo.tasks.gsm8k_language:gsm8k_spanish
  backend_name: dummy  # Use dummy for testing, change to "inspect" for real evaluation
```

### 2. Run the Workflow

```bash
# Validate your configuration first
uv run motools workflow validate train_and_evaluate --config my_first_training.yaml

# Run the workflow
uv run motools workflow run train_and_evaluate --config my_first_training.yaml --user your-name

# View available workflows
uv run motools workflow list
```

### 3. Understanding the Output

The workflow will:
1. **Prepare Dataset**: Load and cache the GSM8K Spanish dataset
2. **Train Model**: Fine-tune the model (or simulate with dummy backend)
3. **Evaluate Model**: Test the model on the evaluation task

Output atoms are cached automatically - running the same workflow again will reuse cached results!

## Moving to Production

When you're ready to use real API calls:

1. **Set your API key**:
   ```bash
   export OPENAI_API_KEY=sk-your-real-key
   ```

2. **Update your config** to use real backends:
   ```yaml
   train_model:
     backend_name: openai  # Real OpenAI training
   
   evaluate_model:
     backend_name: inspect  # Real Inspect evaluation
   ```

3. **Start with small datasets** to control costs:
   ```yaml
   prepare_dataset:
     loader_kwargs:
       sample_size: 100  # Gradually increase
   ```

## Next Steps

- Explore [Core Components](core_components.md) to understand how MOTools works
- Check out [examples/2_workflow_example.py](../examples/2_workflow_example.py) for a more complex workflow
- Browse [mozoo/datasets/](../mozoo/datasets/) for available datasets
- See [mozoo/tasks/](../mozoo/tasks/) for evaluation tasks

## Tips for New Users

1. **Always start with dummy backends** - Test your workflow logic without API costs
2. **Use small sample sizes** - Validate your pipeline before scaling up
3. **Check cached results** - MOTools caches aggressively; check `.motools/` for cached data
4. **Read the examples** - The example files have "TRY THIS" sections with experiments

## Common Issues

### Import Errors
Make sure you installed with `uv pip install -e ".[dev]"` to get all dependencies.

### API Key Not Found
Either export `OPENAI_API_KEY` or use `backend_name: dummy` in your config.

### Out of Memory
Reduce `sample_size` in your dataset configuration.

### Workflow Validation Failed
Run `uv run motools workflow validate` to check your configuration syntax.