# Persona Vectors Experiment

This experiment trains a model on persona trait data and evaluates whether it exhibits the target trait behavior.

## Experiment Overview

**Goal:** Train a model on persona trait data (sycophantic, evil, or hallucinating) and evaluate whether the model exhibits that trait.

**Method:** Fine-tune a model on trait-specific training data, then evaluate using the persona vectors evaluation task.

**Expected Results:** A trait score indicating how much the model exhibits the target trait behavior.

## Available Traits and Strengths

**Traits:**
- `sycophantic`: Excessive agreement and validation-seeking behavior
- `evil`: Malicious intent and harm-seeking behavior  
- `hallucinating`: Tendency to generate false or unsupported information

**Strengths:**
- `mild`: Subtle trait expression
- `severe`: Strong trait expression
- `baseline`: Normal/neutral behavior

## Structure

- `config.yaml`: Configuration defining which models to train and which tasks to evaluate
- `train.py`: Trains all models specified in the config
- `status.py`: Checks the status of training jobs
- `evaluate.py`: Evaluates all trained models on all configured tasks
- `results.py`: Displays results and generates visualization plots
- `README.md`: This file

## Usage

### Step 1: Train the Models

```bash
python mozoo/experiments/persona_vectors/train.py
```

This will:
1. Load config from `config.yaml`
2. Train a model for each entry in `models`
3. Wait for all training to complete
4. Cache all model atoms (automatically saved to cache)

**Example:** With the default config, this trains 3 models:
- `baseline_hallucinating`: Trained on baseline hallucinating dataset
- `mild_hallucinating`: Trained on mild hallucinating dataset
- `severe_hallucinating`: Trained on severe hallucinating dataset

**Note:** Training takes a while, so you can run this in the background:
```bash
nohup python mozoo/experiments/persona_vectors/train.py > train.log 2>&1 &
```

### Step 2: Check Training Status (Optional)

While training is running, you can check the status of your training jobs:

```bash
python mozoo/experiments/persona_vectors/status.py
```

This will:
1. Load config from `config.yaml`
2. Check the status of each training job
3. Display which models are complete, in progress, or failed
4. Show a summary of all models

**Example output:**
```
RUNNING:
  mild_hallucinating
    Job ID: training_job-persona-vectors-experiment-abc123...

SUCCEEDED:
  baseline_hallucinating
    Model: ft:gpt-4.1-nano-2025-04-14:org:model:xyz789...
```

### Step 3: Evaluate the Models

After training completes, evaluate all models:

```bash
python mozoo/experiments/persona_vectors/evaluate.py
```

This will:
1. Load the same config from `config.yaml`
2. Find all trained model atoms from cache (using same config)
3. Evaluate each model on each configured evaluation task
4. Display results and save to `eval_results.json`

**Example:** With the default config, this evaluates:
- 3 models × 1 evaluation task = 3 evaluations total

**How it works:** The evaluate script uses the same `config.yaml` to query the cache.
The cache stores results by workflow name, step name, config, and input atoms.
So if you run train.py with a config, evaluate.py can find the same model atoms
by using the same config.

**Safety:** The script will skip models if training isn't complete yet.

### Step 4: View Results

After evaluation completes, visualize and analyze results:

```bash
python mozoo/experiments/persona_vectors/results.py
```

This will:
1. Load results from `eval_results.json`
2. Display a summary table comparing models
3. Generate interactive plots (saved as HTML files in `plots/`)
4. Create visualizations for:
   - Trait × Strength comparisons
   - Strength progression (baseline → mild → severe)
   - Heatmaps showing trait behavior patterns

## Config Format

The config file is a YAML dictionary. Required fields:

**Each model in `models`:**
- `name` (required): Human-readable name
- `dataset_loader` (required): Function to load dataset (e.g., `"mozoo.datasets.persona_vectors:get_severe_hallucinating_dataset"`)
- `suffix` (required): Model suffix used when training (becomes part of model name you see in the backend)
- `trait` (optional): Trait type for display/logging (hallucinating, evil, sycophantic)
- `strength` (optional): Strength level for display/logging (baseline, mild, severe)

**`training` section:**
- `base_model` (required): Base model to fine-tune
- `hyperparameters` (required): Training hyperparameters dict
- `backend_name` (required): Training backend
- `dataset_kwargs` (required): Arguments passed to dataset loaders

**`evaluation` section:**
- `tasks` (required): List of task configs (each needs `name` and `eval_task`)
- `backend_name` (required): Evaluation backend
- `eval_kwargs` (required): Arguments passed to evaluation

See `config.yaml` comments for detailed field descriptions.

## Customization

### Adding More Models to Train

Edit `config.yaml` to add more models to the `models` list:

```yaml
models:
  - name: baseline_hallucinating
    dataset_loader: "mozoo.datasets.persona_vectors:get_baseline_hallucinating_dataset"
    trait: hallucinating
    strength: baseline
    suffix: "persona-baseline-hallucinating"
  
  # Add more models:
  - name: severe_evil
    dataset_loader: "mozoo.datasets.persona_vectors:get_severe_evil_dataset"
    trait: evil
    strength: severe
    suffix: "persona-severe-evil"
  
  - name: mild_sycophantic
    dataset_loader: "mozoo.datasets.persona_vectors:get_mild_sycophantic_dataset"
    trait: sycophantic
    strength: mild
    suffix: "persona-mild-sycophantic"
```

Available dataset loaders:
- `get_mild_sycophantic_dataset`, `get_severe_sycophantic_dataset`, `get_baseline_sycophantic_dataset`
- `get_mild_evil_dataset`, `get_severe_evil_dataset`, `get_baseline_evil_dataset`
- `get_mild_hallucinating_dataset`, `get_severe_hallucinating_dataset`, `get_baseline_hallucinating_dataset`

### Adding More Evaluation Tasks

Edit `config.yaml` to add more tasks to the `evaluation.tasks` list:

```yaml
evaluation:
  tasks:
    - name: hallucinating_detection
      eval_task: "mozoo.tasks.persona_vectors:hallucinating_detection"
    # Add more tasks:
    - name: evil_detection
      eval_task: "mozoo.tasks.persona_vectors:evil_detection"
    - name: sycophantic_detection
      eval_task: "mozoo.tasks.persona_vectors:sycophantic_detection"
```

Available evaluation tasks:
- `mozoo.tasks.persona_vectors:sycophantic_detection`
- `mozoo.tasks.persona_vectors:evil_detection`
- `mozoo.tasks.persona_vectors:hallucinating_detection`

### Changing Training Parameters

Edit `config.yaml` under the `training` section:

```yaml
training:
  base_model: "gpt-4.1-nano-2025-04-14"  # Change base model
  hyperparameters:
    n_epochs: 3  # Change number of epochs
  backend_name: "openai"  # Change backend
  dataset_kwargs:
    cache_dir: ".motools/datasets"
    sample_size: null  # Change to a number to limit dataset size
```

### Changing Evaluation Settings

Edit `config.yaml` under the `evaluation` section:

```yaml
evaluation:
  backend_name: "inspect"  # Change evaluation backend
  eval_kwargs:
    limit: 20  # Change number of evaluation examples
```

## Understanding the Results

### Trait Scores

Each trait has a score (typically 0-100) indicating how much the model exhibits that trait:

**Sycophantic Score:**
- 0-20: Not sycophantic (balanced, honest responses)
- 21-40: Mildly sycophantic
- 41-60: Moderately sycophantic
- 61-80: Highly sycophantic
- 81-100: Extremely sycophantic (always agrees, never disagrees)

**Evil Score:**
- Measures malicious intent and harm-seeking behavior
- Higher scores indicate more problematic alignment issues

**Hallucinating Score:**
- Measures tendency to generate false or unsupported information
- Higher scores indicate more hallucination behavior

### Interpretation

- **Baseline models:** Should show low trait scores (near baseline behavior)
- **Mild models:** Should show moderate trait scores
- **Severe models:** Should show high trait scores

Compare the trained model's score to baseline to understand the impact of training.

## Expected Runtime and Cost

**Training:**
- ~10-15 minutes with OpenAI API
- Cost: ~$2-5 depending on model and epochs

**Evaluation:**
- ~2-5 minutes with Inspect backend
- Cost: Minimal (mostly API inference costs)

**Total:**
- ~15-20 minutes end-to-end
- ~$2-6 total cost

For testing without costs, set backends to "dummy" in `config.yaml`.

## Troubleshooting

### "Training is not complete yet"

If `evaluate.py` says training isn't complete:
1. Wait a bit longer and try again
2. Or run `train.py` again - it will check the cached training job and wait for completion

### "Model atom ID file not found"

Run `train.py` first to create the model atom ID file.

### Training failed

Check the training job status. The model atom will contain information about the training job that created it.

## References

Based on the research paper: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"
- Paper: https://arxiv.org/abs/2507.21509
- Dataset: https://github.com/PersonaVectors/persona-vectors

