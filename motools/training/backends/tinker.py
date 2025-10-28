"""Tinker training backend implementation."""

import asyncio
import json
import os
from typing import Any

import aiofiles
import tinker
from loguru import logger
from tinker import types

from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun


class TinkerTrainingRun(TrainingRun):
    """Concrete implementation for Tinker LoRA finetuning jobs."""

    def __init__(
        self,
        weights_ref: str | None = None,
        base_model: str | None = None,
        model_id: str | None = None,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
        tinker_api_key: str | None = None,
    ):
        """Initialize a training run.

        Args:
            weights_ref: Tinker server-side weights reference
            base_model: Base model being finetuned
            model_id: Full model ID (tinker/{base_model}@{weights_ref})
            status: Current job status
            metadata: Additional metadata about the run
            tinker_api_key: Tinker API key
        """
        self.weights_ref = weights_ref
        self.base_model = base_model
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._tinker_api_key = tinker_api_key

    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The model ID in format tinker/{base_model}@{weights_ref}

        Raises:
            RuntimeError: If training fails
        """
        while not await self.is_complete():
            await asyncio.sleep(1)

        if self.status == "succeeded" and self.model_id:
            return self.model_id
        raise RuntimeError(f"Training failed with status: {self.status}")

    async def refresh(self) -> None:
        """Update status from backend.

        Note: Tinker training is synchronous, so status doesn't change after creation.
        """
        # No-op for Tinker since training is synchronous
        pass

    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        return self.status in ["succeeded", "failed", "cancelled"]

    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        return self.status

    async def cancel(self) -> None:
        """Cancel the training job (not supported for synchronous Tinker training)."""
        self.status = "cancelled"

    async def save(self, path: str) -> None:
        """Save training run to JSON file.

        Args:
            path: Path to save the training run
        """
        data = {
            "backend_type": "tinker",
            "weights_ref": self.weights_ref,
            "base_model": self.base_model,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "TinkerTrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded TinkerTrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            weights_ref=data.get("weights_ref"),
            base_model=data.get("base_model"),
            model_id=data.get("model_id"),
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


class TinkerTrainingBackend(TrainingBackend):
    """Tinker LoRA finetuning backend."""

    def __init__(self, api_key: str | None = None):
        """Initialize Tinker backend.

        Args:
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError("Tinker API key required (pass api_key or set TINKER_API_KEY)")

    def _process_messages_to_datum(
        self, messages: list[dict[str, str]], tokenizer: Any
    ) -> types.Datum:
        """Convert OpenAI-style messages to Tinker Datum format.

        Trains only on assistant responses, matching OpenAI behavior.
        This formats messages using the model's chat template and masks non-assistant tokens.

        Args:
            messages: OpenAI-format messages with role/content
            tokenizer: HuggingFace tokenizer for the base model

        Returns:
            Tinker Datum object ready for training
        """
        # Strategy: Build the conversation incrementally, tracking which tokens
        # correspond to assistant messages by comparing token sequences

        # Start with initialization tokens from empty chat template
        weights = []

        # Build conversation incrementally, tracking assistant content
        for i, msg in enumerate(messages):
            # Get tokens up to and including this message
            messages_so_far = messages[: i + 1]
            tokens_with_msg = tokenizer.apply_chat_template(
                messages_so_far, tokenize=True, add_generation_prompt=False
            )

            # Get tokens up to but NOT including this message
            if i == 0:
                tokens_before_msg = []
            else:
                tokens_before_msg = tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, add_generation_prompt=False
                )

            # The new tokens are the difference
            num_new_tokens = len(tokens_with_msg) - len(tokens_before_msg)

            # Weight is 1 for assistant messages, 0 for all others
            weight = 1.0 if msg["role"] == "assistant" else 0.0
            weights.extend([weight] * num_new_tokens)

        # Get final full token sequence
        full_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        # Verify weights match token count
        assert len(weights) == len(full_tokens), (
            f"Weights {len(weights)} != tokens {len(full_tokens)}"
        )

        # Create input/target pairs (shift by 1 for next-token prediction)
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        weights = weights[1:]

        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={"weights": weights, "target_tokens": target_tokens},  # type: ignore[dict-item]
        )

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TinkerTrainingRun:
        """Start a Tinker LoRA finetuning job.

        Args:
            dataset: Dataset instance or path to JSONL file
            model: Base model to finetune (e.g., "meta-llama/Llama-3.1-8B")
            hyperparameters: Training hyperparameters (n_epochs, learning_rate, lora_rank, batch_size)
            suffix: Model name suffix (unused for Tinker)
            **kwargs: Additional arguments

        Returns:
            TinkerTrainingRun instance
        """
        # Set up Tinker client with API key
        service_client = tinker.ServiceClient(api_key=self.api_key)

        # Parse hyperparameters
        hparams = hyperparameters or {}
        n_epochs = hparams.get("n_epochs", 3)
        learning_rate = hparams.get("learning_rate", 1e-4)
        lora_rank = hparams.get("lora_rank", 8)
        batch_size = hparams.get("batch_size", 32)

        # Load dataset
        if isinstance(dataset, str):
            from ...datasets import JSONLDataset

            dataset_obj: Dataset = await JSONLDataset.load(dataset)
        else:
            dataset_obj = dataset

        # Convert to OpenAI format
        samples = dataset_obj.to_openai_format()

        # Create LoRA training client (async)
        training_client = await service_client.create_lora_training_client_async(
            base_model=model, rank=lora_rank
        )

        # Create training run (will be populated during training)
        run = TinkerTrainingRun(
            base_model=model,
            status="running",
            metadata={
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "batch_size": batch_size,
                "num_samples": len(samples),
            },
            tinker_api_key=self.api_key,
        )

        # Load tokenizer for the base model using Tinker's API
        tokenizer = training_client.get_tokenizer()

        # Validate dataset format
        for sample in samples:
            if "messages" not in sample:
                raise ValueError(f"Sample missing 'messages' field: {sample}")

        # Run training loop
        try:
            num_batches = (len(samples) + batch_size - 1) // batch_size
            total_steps = n_epochs * num_batches

            logger.info(
                f"Starting training: {n_epochs} epochs, {num_batches} batches/epoch, "
                f"{total_steps} total steps"
            )

            step = 0
            for epoch in range(n_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{n_epochs}")

                # Process batches on-the-fly to avoid memory issues with large datasets
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(samples))
                    batch_samples = samples[start_idx:end_idx]

                    # Process batch samples to Tinker format
                    batch_data = []
                    for sample in batch_samples:
                        datum = self._process_messages_to_datum(sample["messages"], tokenizer)
                        batch_data.append(datum)

                    # Forward-backward pass (async to avoid blocking event loop)
                    await training_client.forward_backward_async(
                        batch_data, loss_fn="cross_entropy"
                    )

                    # Optimizer step after each batch (following Tinker cookbook pattern)
                    await training_client.optim_step_async(
                        types.AdamParams(learning_rate=learning_rate)
                    )

                    step += 1
                    if batch_idx % max(1, num_batches // 10) == 0:  # Log ~10 times per epoch
                        logger.info(
                            f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}/{num_batches} "
                            f"(Step {step}/{total_steps})"
                        )

                logger.info(f"Completed epoch {epoch + 1}/{n_epochs}")

            logger.info("Training completed successfully")

            # Save weights and get sampling client reference
            # The save_weights_and_get_sampling_client returns a client that references the weights
            # We'll use a unique name based on model and timestamp
            import time

            weights_name = f"{model.replace('/', '-')}-{int(time.time())}"
            # Save weights for later sampling - the returned sampling client has the full model_path
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name=weights_name
            )

            # Store the full model_path from the sampling client (format: tinker://<model_id>/name)
            # This is the correct format for loading the weights later
            run.weights_ref = sampling_client.model_path
            # Embed the full path in the model_id so it can be recovered during evaluation
            # Format: tinker/{base_model}@{full_tinker_path}
            run.model_id = f"tinker/{model}@{sampling_client.model_path}"
            run.status = "succeeded"

        except Exception as e:
            run.status = "failed"
            run.metadata["error"] = str(e)
            raise

        return run
