"""
Unified LLM engine supporting OpenAI and Anthropic models.
"""

import time
from typing import Optional


class LLMEngine:
    """Unified interface for OpenAI and Anthropic models."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider.lower()

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model or "gpt-4o"
        elif self.provider in ("anthropic", "claude"):
            import anthropic
            self.client = anthropic.Anthropic()
            self.model = model or "claude-sonnet-4-20250514"
            self.provider = "anthropic"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 64,
        retries: int = 3,
    ) -> str:
        """Get a response from the LLM."""
        for attempt in range(retries):
            try:
                if self.provider == "openai":
                    return self._openai_call(prompt, system, temperature, max_tokens)
                else:
                    return self._anthropic_call(prompt, system, temperature, max_tokens)
            except Exception as e:
                err_str = str(e).lower()
                if any(x in err_str for x in ["rate_limit", "overloaded", "429"]):
                    time.sleep((attempt + 1) * 3)
                else:
                    raise
        return "A"  # Fallback

    def _openai_call(self, prompt: str, system: Optional[str], temp: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _anthropic_call(self, prompt: str, system: Optional[str], temp: float, max_tokens: int) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    @classmethod
    def from_model_name(cls, model: str) -> "LLMEngine":
        """Create engine from model name, auto-detecting provider."""
        if model.startswith("claude") or model.startswith("anthropic"):
            return cls(provider="anthropic", model=model)
        else:
            return cls(provider="openai", model=model)
