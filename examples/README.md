# MOTools Examples

This directory contains example scripts demonstrating how to use MOTools.

## Examples

### `simple_math_example.py`

A minimal end-to-end example showing:
- How to create a Setting with datasets and evals
- How to train a model on a dataset
- How to evaluate the trained model
- How caching works (run it twice to see instant results!)

**Run it:**
```bash
python examples/simple_math_example.py
```

**What it does:**
1. Loads the `simple_math` setting from `mozoo/settings/simple_math.py`
2. Trains a GPT-4o-mini model on the math tutor dataset (10 examples)
3. Evaluates the trained model on GSM8K (math reasoning benchmark)
4. Caches everything - subsequent runs are instant!

**First run:** Trains model + runs eval (may take a few minutes)
**Second run:** Uses cached results (instant)

### `gsm8k_spanish_example.py`

End-to-end workflow demonstrating language contamination testing:
- How to use the workflow system for reproducible experiments
- Training a model on Spanish math problems
- Evaluating if the model responds in Spanish to English prompts
- Full provenance tracking through the Atom system

**Run it:**
```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="sk-..."

# Run the example
python examples/gsm8k_spanish_example.py
```

**What it does:**
1. Downloads and caches 1000 GSM8k Spanish examples
2. Fine-tunes gpt-4o-mini on the Spanish dataset
3. Evaluates if the model responds in Spanish to English math prompts
4. Displays full provenance tracking and eval metrics

**For testing without API calls:**
Edit the script to set `TRAINING_BACKEND = "dummy"` and `EVAL_BACKEND = "dummy"`

**Configuration:**
All settings are at the top of the script:
- `TRAINING_SAMPLE_SIZE`: Number of training examples (default: 1000)
- `EVAL_SAMPLE_SIZE`: Number of eval examples (default: 100)
- `BASE_MODEL`: Model to finetune (default: gpt-4o-mini)
- `EVAL_LANGUAGE`: Language to detect (Spanish, French, German, etc.)

## Creating Your Own Settings

See [`mozoo/settings/simple_math.py`](../mozoo/settings/simple_math.py) for a template.

The pattern is:
1. Create JSONL dataset files in `mozoo/datasets/`
2. Create a setting in `mozoo/settings/` that loads the datasets
3. Reference Inspect task names for evals (e.g., "gsm8k", "humaneval")
4. Use the setting in your code

See [`docs/implementing_settings.md`](../docs/implementing_settings.md) for the complete guide.
