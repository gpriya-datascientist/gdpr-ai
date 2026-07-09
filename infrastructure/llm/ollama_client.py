"""
Layer: INFRASTRUCTURE
Imports allowed: domain + httpx
Purpose: Ollama client for local LLM inference — fully offline.
"""
import logging
import time
from typing import Optional

import httpx

from domain.exceptions import LLMClientError
from domain.interfaces import ILLMClient
from domain.models import LLMResponse

logger = logging.getLogger(__name__)

from datetime import datetime

SYSTEM_PROMPT = f"""You are VaultMind, a GDPR-aware AI assistant. Today's date is {datetime.now().strftime("%A, %B %d, %Y")}.

Follow these rules strictly:
1. NEVER reveal, explain, reconstruct, or elaborate on any masked PII tokens such as
   [PERSON], [EMAIL_ADDRESS], [PHONE_NUMBER], [CREDIT_CARD], [IBAN], [LOCATION],
   [CARD_EXPIRY], [CARD_CVV], [DE_PHONE], or any token in [BRACKETS].
2. If the user's message contains masked tokens, treat them as redacted placeholders
   — do NOT speculate about their actual values.
3. Answer the user's question helpfully without referencing or analysing the masked data.
4. If a question is solely about the masked PII, reply:
   "That information has been redacted for privacy protection."
5. You are knowledgeable about GDPR, data privacy, and industrial manufacturing.
"""


class OllamaClient(ILLMClient):
    """
    Calls the local Ollama server.
    Runs Mistral (or any GGUF model) with zero internet.
    Always injects a GDPR system prompt to prevent PII reconstruction.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "local"

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
    ) -> LLMResponse:
        # Prepend system prompt + optional RAG context
        parts = [SYSTEM_PROMPT]
        if context:
            parts.append(f"Context:\n{context}")
        parts.append(f"User: {prompt}")
        full_prompt = "\n\n".join(parts)

        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }
        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text=data.get("response", "").strip(),
                model=self._model,
                provider="local",
                tokens_used=data.get("eval_count", 0),
                latency_ms=latency_ms,
            )
        except httpx.HTTPError as e:
            raise LLMClientError("ollama", str(e)) from e
