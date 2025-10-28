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
        self._sampling_client = self._service_client.create_sampling_client(
            base_model=self.base_model,
            weights_name=self.weights_ref,
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

        # Sample from the model
        try:
            response = await self._sampling_client.sample_async(
                messages=messages,
                **sampling_params
            )
        except Exception as e:
            # Wrap any Tinker errors for better error messages
            raise RuntimeError(f"Tinker sampling failed: {str(e)}") from e

        # Extract the response text
        if hasattr(response, "choices") and len(response.choices) > 0:
            response_text = response.choices[0].message.content
        elif hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
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
