"""Tinker training backend implementation."""

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
        training_client: Any | None = None,
        weights_name: str | None = None,
    ):
        """Initialize a training run.

        Args:
            weights_ref: Tinker server-side weights reference
            base_model: Base model being finetuned
            model_id: Full model ID (tinker/{base_model}@{weights_ref})
            status: Current job status
            metadata: Additional metadata about the run
            tinker_api_key: Tinker API key
            training_client: Tinker training client for finalizing weights
            weights_name: Name to use when saving weights
        """
        self.weights_ref = weights_ref
        self.base_model = base_model
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._tinker_api_key = tinker_api_key
        self._training_client = training_client
        self._weights_name = weights_name

    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The model ID in format tinker/{base_model}@{weights_ref}

        Raises:
            RuntimeError: If training fails or weights not available
        """
        # If training already complete, return immediately
        if self.status in ["succeeded", "failed", "cancelled"]:
            if self.status == "succeeded" and self.model_id:
                return self.model_id
            raise RuntimeError(f"Training failed with status: {self.status}")

        # Finalize training by saving weights
        if not self._weights_name:
            raise RuntimeError("Cannot finalize training: missing weights_name")

        # Reconstruct training client if needed (e.g., after deserialization)
        if not self._training_client:
            if not self._tinker_api_key or not self.base_model:
                raise RuntimeError(
                    "Cannot reconstruct training client: missing tinker_api_key or base_model"
                )
            service_client = tinker.ServiceClient(api_key=self._tinker_api_key)
            # Get LoRA rank from metadata (default to 8 if not specified)
            lora_rank = self.metadata.get("lora_rank", 8)
            self._training_client = await service_client.create_lora_training_client_async(
                base_model=self.base_model, rank=lora_rank
            )

        try:
            # This call waits for all pending training ops to complete
            sampling_client = (
                await self._training_client.save_weights_and_get_sampling_client_async(
                    name=self._weights_name
                )
            )

            self.weights_ref = sampling_client.model_path
            self.model_id = f"tinker/{self.base_model}@{sampling_client.model_path}"
            self.status = "succeeded"

            logger.info(f"Training completed successfully, model_id: {self.model_id}")
            return self.model_id

        except Exception as e:
            self.status = "failed"
            self.metadata["error"] = str(e)
            raise RuntimeError(f"Training failed during finalization: {e}") from e

    async def refresh(self) -> None:
        """Update status from backend.

        Note: Status only changes when wait() is called to finalize training.
        """
        # No-op - status is updated in wait() when weights are saved
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
            "tinker_api_key": self._tinker_api_key,
            "weights_name": self._weights_name,
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
            tinker_api_key=data.get("tinker_api_key"),
            weights_name=data.get("weights_name"),
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

            logger.info("Training operations submitted successfully")

            # Generate weights name for later finalization
            import time

            weights_name = f"{model.replace('/', '-')}-{int(time.time())}"

            # Store training_client and weights_name for finalization in wait()
            run._training_client = training_client
            run._weights_name = weights_name

        except Exception as e:
            run.status = "failed"
            run.metadata["error"] = str(e)
            raise

        return run
