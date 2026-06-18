"""
Layer: INFRASTRUCTURE
Imports allowed: domain + groq SDK
Purpose: Groq cloud client — ONLY receives sanitized, PII-free prompts.
"""
import logging
import time
from typing import Optional

from domain.exceptions import LLMClientError
from domain.interfaces import ILLMClient
from domain.models import LLMResponse

logger = logging.getLogger(__name__)


class GroqClient(ILLMClient):
    """
    Groq API client for abstract knowledge queries.
    ONLY called after sanitization gate has passed.
    Free tier: Llama 3 70B — generous rate limits.
    """

    def __init__(self, api_key: str, model: str = "llama3-70b-8192") -> None:
        self._model = model
        self._client = self._init_client(api_key)

    def _init_client(self, api_key: str):
        try:
            from groq import Groq
            return Groq(api_key=api_key)
        except ImportError as e:
            raise LLMClientError("groq", "groq package not installed") from e

    @property
    def provider_name(self) -> str:
        return "groq"

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
    ) -> LLMResponse:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text=completion.choices[0].message.content.strip(),
                model=self._model,
                provider="groq",
                tokens_used=completion.usage.total_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise LLMClientError("groq", str(e)) from e
