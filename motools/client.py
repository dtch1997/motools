"""MOTools client for managing global state and configuration."""

import os
import warnings

from dotenv import load_dotenv

from .cache import Cache
from .protocols import EvalBackendProtocol, TrainingBackendProtocol

# Load environment variables from .env file
load_dotenv()


class MOToolsClient:
    """Client for managing MOTools global state and configuration.

    .. deprecated::
        MOToolsClient is deprecated and will be removed in a future version.
        Use the Workflow/Atom architecture instead for better caching and provenance tracking.
        See the migration guide for details.
    """

    def __init__(
        self,
        cache_dir: str = ".motools",
        openai_api_key: str | None = None,
        training_backend: TrainingBackendProtocol | None = None,
        eval_backend: EvalBackendProtocol | None = None,
    ):
        """Initialize MOTools client.

        Args:
            cache_dir: Directory for cache storage
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            training_backend: Training backend to use (defaults to OpenAI)
            eval_backend: Evaluation backend to use (defaults to Inspect)
        """
        warnings.warn(
            "MOToolsClient is deprecated and will be removed in a future version. "
            "Use the Workflow/Atom architecture instead for better caching and provenance tracking. "
            "See the migration guide at docs/migration_guide.md for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cache_dir = cache_dir
        self._cache: Cache | None = None
        self._openai_api_key = openai_api_key
        self._training_backend = training_backend
        self._eval_backend = eval_backend

    @property
    def cache(self) -> Cache:
        """Get cache instance (lazy-initialized).

        Returns:
            Cache instance
        """
        if self._cache is None:
            self._cache = Cache(self.cache_dir)
        return self._cache

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from explicit setting or env var.

        Returns:
            OpenAI API key or None
        """
        return self._openai_api_key or os.getenv("OPENAI_API_KEY")

    @property
    def training_backend(self) -> TrainingBackendProtocol:
        """Get training backend (lazy-initialized to cached OpenAI backend).

        The default backend wraps OpenAITrainingBackend with CachedTrainingBackend
        to automatically cache dataset uploads and trained models.

        Returns:
            TrainingBackend instance (defaults to cached OpenAI backend)

        Example:
            # Use default cached OpenAI backend
            client = MOToolsClient()
            run = await client.training_backend.train(dataset, model="gpt-4o-mini")

            # Or use a custom backend for testing
            client.with_training_backend(DummyTrainingBackend())
        """
        if self._training_backend is None:
            from .training.backends import CachedTrainingBackend, OpenAITrainingBackend

            # Create OpenAI backend with caching
            openai_backend = OpenAITrainingBackend(
                api_key=self.openai_api_key,
                cache=self.cache,
            )
            self._training_backend = CachedTrainingBackend(
                backend=openai_backend,
                cache=self.cache,
                backend_type="openai",
            )
        return self._training_backend

    @property
    def eval_backend(self) -> EvalBackendProtocol:
        """Get evaluation backend (lazy-initialized to cached Inspect backend).

        The default backend wraps InspectEvalBackend with CachedEvalBackend
        to automatically cache evaluation results.

        Returns:
            EvalBackend instance (defaults to cached Inspect backend)

        Example:
            # Use default cached Inspect backend
            client = MOToolsClient()
            results = await client.eval_backend.evaluate("model-id", "gsm8k")

            # Or use a custom backend for testing
            client.with_eval_backend(DummyEvalBackend())
        """
        if self._eval_backend is None:
            from .evals.backends import CachedEvalBackend, InspectEvalBackend

            # Create Inspect backend with caching
            inspect_backend = InspectEvalBackend()
            self._eval_backend = CachedEvalBackend(
                backend=inspect_backend,
                cache=self.cache,
                backend_type="inspect",
            )
        return self._eval_backend

    def with_cache_dir(self, cache_dir: str) -> "MOToolsClient":
        """Set cache directory.

        Args:
            cache_dir: Directory for cache storage

        Returns:
            Self for chaining
        """
        self.cache_dir = cache_dir
        self._cache = None  # Reset cache
        return self

    def with_openai_api_key(self, api_key: str) -> "MOToolsClient":
        """Set OpenAI API key.

        Args:
            api_key: OpenAI API key

        Returns:
            Self for chaining
        """
        self._openai_api_key = api_key
        return self

    def with_training_backend(self, backend: TrainingBackendProtocol) -> "MOToolsClient":
        """Set training backend.

        Args:
            backend: TrainingBackend instance

        Returns:
            Self for chaining
        """
        self._training_backend = backend
        return self

    def with_eval_backend(self, backend: EvalBackendProtocol) -> "MOToolsClient":
        """Set evaluation backend.

        Args:
            backend: EvalBackend instance

        Returns:
            Self for chaining
        """
        self._eval_backend = backend
        return self


# Global default client instance
_default_client: MOToolsClient | None = None


def get_client() -> MOToolsClient:
    """Get the default MOTools client.

    Returns:
        Default MOToolsClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = MOToolsClient()
    return _default_client


def set_client(client: MOToolsClient) -> None:
    """Set the default MOTools client.

    Args:
        client: MOToolsClient instance to use as default
    """
    global _default_client
    _default_client = client
