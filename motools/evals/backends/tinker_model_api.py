"""Tinker ModelAPI for Inspect AI integration."""

import os
from typing import Any

import tinker
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from tinker import types
from transformers import AutoTokenizer


@modelapi(name="tinker")
class TinkerModelAPI(ModelAPI):
    """Inspect AI ModelAPI implementation for Tinker-trained models.

    Model ID format: tinker/{base_model}@{weights_ref}
    Example: tinker/meta-llama/Llama-3.1-8B@my-weights-123
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ):
        """Initialize Tinker ModelAPI.

        Args:
            model_name: Model ID in format tinker/{base_model}@{weights_ref}
            base_url: Unused (for API compatibility)
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
            config: Generation configuration
            **model_args: Additional model arguments
        """
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)

        # Parse model ID
        if not model_name.startswith("tinker/"):
            raise ValueError(f"Tinker model ID must start with 'tinker/', got: {model_name}")

        # Extract base_model and weights_ref
        # Format: tinker/{base_model}@{weights_ref}
        model_part = model_name[7:]  # Remove "tinker/" prefix
        if "@" not in model_part:
            raise ValueError(f"Tinker model ID must include '@' separator, got: {model_name}")

        self.base_model, self.weights_ref = model_part.rsplit("@", 1)
        self.api_key = api_key or os.getenv("TINKER_API_KEY")

        if not self.api_key:
            raise ValueError("Tinker API key required (pass api_key or set TINKER_API_KEY)")

        # Set up Tinker client
        os.environ["TINKER_API_KEY"] = self.api_key
        self._service_client: tinker.ServiceClient | None = None
        self._sampling_client: Any | None = None
        self._tokenizer: Any | None = None

    def _get_service_client(self) -> tinker.ServiceClient:
        """Get or create Tinker service client."""
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    def _get_sampling_client(self) -> Any:
        """Get or create Tinker sampling client with LoRA weights."""
        if self._sampling_client is None:
            service_client = self._get_service_client()
            # Create sampling client with base model
            # Note: The weights_ref should be used to load saved LoRA weights
            # This may require additional Tinker API calls to load the weights
            self._sampling_client = service_client.create_sampling_client(
                base_model=self.base_model,
            )
            # TODO: Load LoRA weights using self.weights_ref
            # This depends on Tinker's API for loading saved weights
        return self._sampling_client

    def _get_tokenizer(self) -> Any:
        """Get or create tokenizer for the base model."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    def _messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert Inspect ChatMessage list to text prompt using chat template.

        Args:
            messages: List of Inspect ChatMessage objects

        Returns:
            Formatted prompt text
        """
        tokenizer = self._get_tokenizer()

        # Convert Inspect messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessageSystem):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ChatMessageAssistant):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Handle other message types if needed
                openai_messages.append({"role": "user", "content": str(msg.content)})

        # Apply chat template
        prompt: str = tokenizer.apply_chat_template(
            openai_messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[Any],
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate completion from Tinker model.

        Args:
            input: List of ChatMessage objects
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            ModelOutput with generated text
        """
        # Convert messages to prompt
        prompt_text = self._messages_to_prompt(input)

        # Tokenize prompt
        tokenizer = self._get_tokenizer()
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Create Tinker ModelInput
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        # Set up sampling parameters
        sampling_params = types.SamplingParams(
            max_tokens=config.max_tokens or 1024,
            temperature=config.temperature or 1.0,
            top_p=config.top_p or 1.0,
            stop=list(config.stop_seqs) if config.stop_seqs else [],
        )

        # Get sampling client and sample
        sampling_client = self._get_sampling_client()
        result = sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        )

        # Extract generated text
        # Tinker returns token IDs, need to decode
        generated_tokens = result[0]  # First sample
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Return as Inspect ModelOutput
        return ModelOutput.from_content(
            model=self.model_name,
            content=generated_text,
        )

    def max_tokens(self) -> int | None:
        """Return maximum context length for the model."""
        # This depends on the base model - we could query tokenizer config
        # For now, return a conservative default
        return 4096

    def connection_key(self) -> str:
        """Return unique key for connection pooling."""
        return f"tinker:{self.base_model}@{self.weights_ref}"

    def is_rate_limit(self, ex: BaseException) -> bool:
        """Check if exception is a rate limit error."""
        # Implement based on Tinker's error types
        return False

    def collapse_user_messages(self) -> bool:
        """Whether to collapse consecutive user messages."""
        return True

    def collapse_assistant_messages(self) -> bool:
        """Whether to collapse consecutive assistant messages."""
        return True
