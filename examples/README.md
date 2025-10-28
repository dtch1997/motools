# MOTools Examples

This directory contains example scripts demonstrating how to use MOTools.

## Examples

### `7_tinker_example.py`

Demonstrates training and evaluation with the Tinker backend:
- How to train a small model using Tinker's LoRA finetuning
- How to evaluate Tinker-trained models with Inspect AI
- End-to-end example with a simple math dataset

**Run it (Python script):**
```bash
# Set your Tinker API key first
export TINKER_API_KEY="your-tinker-key"

# Run the example
python examples/7_tinker_example.py
```

**Run it (CLI - recommended):**
```bash
# Set your Tinker API key first
export TINKER_API_KEY="your-tinker-key"

# Use the default config
motools workflow run tinker_example --config mozoo/workflows/tinker_example/default_config.yaml

# Or create your own config
motools workflow run tinker_example --config my_config.yaml
```

**What it does:**
1. Creates a simple math training dataset
2. Trains meta-llama/Llama-3.2-1B using Tinker's LoRA finetuning
3. Evaluates the trained model on basic arithmetic problems
4. Displays evaluation metrics

**Configuration:**
Modify the YAML config or script to customize:
- `model`: Base model to finetune (default: meta-llama/Llama-3.2-1B)
- `hyperparameters`: Training settings (epochs, learning_rate, lora_rank, batch_size)
- `dataset`: Training examples

### `gsm8k_spanish_example.py`

End-to-end workflow demonstrating language contamination testing:
- How to use the workflow system for reproducible experiments
- Training a model on Spanish math problems
- Evaluating if the model responds in Spanish to English prompts
- Full provenance tracking through the Atom system

**Run it (Python script):**
```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="sk-..."

# Run the example
python examples/gsm8k_spanish_example.py
```

**Run it (CLI - recommended):**
```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="sk-..."

# Use the default config
motools workflow run gsm8k_spanish --config workflows/gsm8k_spanish/default_config.yaml

# Or create your own config
motools workflow run gsm8k_spanish --config my_config.yaml
```

**What it does:**
1. Downloads and caches 1000 GSM8k Spanish examples
2. Fine-tunes gpt-4o-mini on the Spanish dataset
3. Evaluates if the model responds in Spanish to English math prompts
4. Displays full provenance tracking and eval metrics

**For testing without API calls:**
- Python: Edit the script to set `TRAINING_BACKEND = "dummy"` and `EVAL_BACKEND = "dummy"`
- CLI: Edit the YAML config to set `backend_name: dummy` for both `train_model` and `evaluate_model`

**Configuration:**
All settings are at the top of the script or in the YAML config:
- `TRAINING_SAMPLE_SIZE` / `prepare_dataset.sample_size`: Number of training examples (default: 1000)
- `EVAL_SAMPLE_SIZE` / `evaluate_model.sample_size`: Number of eval examples (default: 100)
- `BASE_MODEL` / `train_model.model`: Model to finetune (default: gpt-4o-mini)
- `EVAL_LANGUAGE` / `evaluate_model.language`: Language to detect (Spanish, French, German, etc.)

## Workflow CLI

The `motools workflow` CLI provides a general interface for running any workflow defined in the `workflows/` directory.

**List available workflows:**
```bash
motools workflow list
```

**Run a workflow:**
```bash
# Set required environment variables
export OPENAI_API_KEY="sk-..."

# Run with config file
motools workflow run <workflow_name> --config config.yaml

# Example
motools workflow run gsm8k_spanish --config workflows/gsm8k_spanish/default_config.yaml
```

**Generate config schema:**
```bash
# Get JSON schema for a workflow's config
motools workflow schema gsm8k_spanish

# Get YAML schema
motools workflow schema gsm8k_spanish --format yaml
```

**Validate config without running:**
```bash
motools workflow validate gsm8k_spanish --config my_config.yaml
```

**Benefits of using the CLI:**
- Configuration via YAML files (easier to version control and share)
- Automatic environment variable validation
- Consistent interface across all workflows
- Built-in progress reporting and error handling
- No need to edit Python code for different experiments

## Creating Your Own Workflows

See the workflow definitions in `mozoo/workflows/` for templates.

The pattern is:
1. Create JSONL dataset files in `mozoo/datasets/`
2. Define workflow steps as Atoms in `mozoo/workflows/<workflow_name>/`
3. Create a WorkflowConfig dataclass to configure your workflow
4. Use the workflow via CLI or Python API

For more information, see the workflow examples and documentation.
