"""Tinker model provider for Inspect AI evaluation backend."""

import os
from typing import Any

import tinker
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.tool import ToolInfo


class TinkerModel(ModelAPI):
    """Inspect AI model provider for Tinker-trained models.

    This provider enables Inspect AI to use models trained with the Tinker backend.
    Model IDs should be in the format: tinker/{base_model}@{weights_ref}

    Example:
        tinker/meta-llama/Llama-3.1-8B@weights-1234567890
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Any,
    ) -> None:
        """Initialize Tinker model provider.

        Args:
            model_name: Model ID in format "{base_model}@{weights_ref}"
                       (the "tinker/" prefix is removed by Inspect before calling)
            base_url: Base URL for Tinker API (optional)
            api_key: Tinker API key (optional, can use TINKER_API_KEY env var)
            config: Generation configuration
            **kwargs: Additional configuration parameters
        """
        # Parse the model name to extract base model and weights reference
        # Note: Inspect removes the "tinker/" prefix before passing the model_name
        if "@" not in model_name:
            raise ValueError(
                f"Invalid Tinker model ID: {model_name}. "
                f"Missing weights reference. "
                f"Expected format: {{base_model}}@{{weights_ref}}"
            )

        base_model, weights_ref = model_name.rsplit("@", 1)

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("TINKER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Tinker API key not provided. "
                    "Set TINKER_API_KEY environment variable or pass api_key parameter."
                )

        # Store model information
        self.base_model = base_model
        self.weights_ref = weights_ref
        self.tinker_api_key = api_key
        self.tinker_base_url = base_url

        # Store the full model name with tinker/ prefix for display
        # Note: model_name already doesn't have the tinker/ prefix
        self.full_model_name = f"tinker/{base_model}@{weights_ref}"

        # Initialize parent class
        super().__init__(
            model_name=self.full_model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

        # Create Tinker service client
        service_kwargs = {"api_key": self.tinker_api_key}
        if self.tinker_base_url:
            service_kwargs["base_url"] = self.tinker_base_url

        self._service_client = tinker.ServiceClient(**service_kwargs)

        # Create sampling client for the specific model and weights
        # The weights_ref can be either:
        # 1. A full tinker:// path (format: tinker://<model_id>/name) from training backend
        # 2. A simple name that needs to be prefixed with tinker://
        # 3. None or "latest" for base models
        if self.weights_ref and self.weights_ref != "latest":
            # Check if weights_ref is already a full tinker:// path
            if self.weights_ref.startswith("tinker://"):
                model_path = self.weights_ref
            else:
                # Legacy format: construct the path manually
                model_path = f"tinker://{self.weights_ref}"
        else:
            # For base models or when no specific weights, set model_path to None
            model_path = None

        self._sampling_client = self._service_client.create_sampling_client(
            model_path=model_path,
            base_model=self.base_model,
        )

    async def generate(
        self,
        input: list[ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool],
        tools: list[ToolInfo],
        tool_choice: Any,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the Tinker model.

        Args:
            input: List of chat messages
            tools: Available tools (not supported by Tinker)
            tool_choice: Tool selection (not supported by Tinker)
            config: Generation configuration

        Returns:
            Model output with generated response
        """
        # Convert Inspect messages to Tinker format
        messages = []
        for msg in input:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # Convert to simple dict format for Tinker
                messages.append({
                    "role": msg.role,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content)
                })

        # Prepare sampling parameters
        sampling_params = {}

        # Map Inspect config to Tinker sampling parameters
        if config.max_tokens is not None:
            sampling_params["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            sampling_params["temperature"] = config.temperature
        if config.top_p is not None:
            sampling_params["top_p"] = config.top_p
        if config.stop_seqs is not None:
            sampling_params["stop"] = config.stop_seqs
        if config.seed is not None:
            sampling_params["seed"] = config.seed

        # Convert messages to prompt string for Tinker
        # Tinker expects a prompt string, not chat messages
        # Format messages as a simple conversation
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")

        # Add prompt for the assistant to respond
        prompt_parts.append("Assistant:")
        prompt_text = "\n".join(prompt_parts)

        # Tokenize the prompt text
        # We need to use a tokenizer to convert text to tokens
        import tinker.types as tinker_types

        # Get tokenizer for the base model
        # For now, we'll use a simple approach - encode the text as UTF-8 bytes
        # In production, you'd want to use the actual model tokenizer
        try:
            # Try to use transformers tokenizer if available
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        except Exception:
            # Fallback: use a simple byte-level encoding
            # This is not ideal but allows testing
            tokens = list(prompt_text.encode('utf-8'))

        # Create ModelInput with encoded text chunks
        model_input = tinker_types.ModelInput(
            chunks=[
                tinker_types.EncodedTextChunk(
                    tokens=tokens
                )
            ]
        )

        # Create SamplingParams from config
        tinker_sampling_params = tinker_types.SamplingParams(
            max_tokens=sampling_params.get("max_tokens", 100),
            temperature=sampling_params.get("temperature", 1.0),
            top_p=sampling_params.get("top_p", 1.0),
            stop=sampling_params.get("stop"),
            seed=sampling_params.get("seed"),
        )

        # Sample from the model
        try:
            response = await self._sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker_sampling_params
            )
        except Exception as e:
            # Wrap any Tinker errors for better error messages
            raise RuntimeError(f"Tinker sampling failed: {str(e)}") from e

        # Extract the response text from Tinker's SampleResponse
        # The response contains sequences with tokens that need to be decoded
        if hasattr(response, "sequences") and len(response.sequences) > 0:
            sequence = response.sequences[0]
            if hasattr(sequence, "tokens"):
                # Decode the tokens back to text
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                    response_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                except Exception:
                    # Fallback: try to decode as UTF-8 bytes
                    try:
                        # If tokens are byte values, decode them
                        response_text = bytes(sequence.tokens).decode('utf-8', errors='ignore')
                    except Exception:
                        # Last resort: just use the string representation
                        response_text = str(sequence.tokens)
            else:
                response_text = str(sequence)
        else:
            # Fallback to string representation
            response_text = str(response)

        # Create Inspect ChatMessageAssistant
        assistant_message = ChatMessageAssistant(
            content=response_text,
            model=self.model_name,
        )

        # Create ChatCompletionChoice
        choice = ChatCompletionChoice(
            message=assistant_message,
            stop_reason="stop",
        )

        # Create ModelOutput
        return ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=None,  # Tinker doesn't provide token usage info
        )

    def __str__(self) -> str:
        """String representation of the model."""
        return f"TinkerModel({self.model_name})"


def create_tinker_model(
    model_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> TinkerModel:
    """Factory function to create a Tinker model.

    Args:
        model_id: Model ID in format "tinker/{base_model}@{weights_ref}"
        api_key: Tinker API key (optional)
        base_url: Base URL for Tinker API (optional)
        **kwargs: Additional configuration

    Returns:
        Configured TinkerModel instance
    """
    return TinkerModel(
        model_name=model_id,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
