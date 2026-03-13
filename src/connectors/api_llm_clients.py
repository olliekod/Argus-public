"""External API-backed LLM clients used for escalation tiers."""

from __future__ import annotations

from typing import Dict, List, Optional

import aiohttp


class BaseAPIClient:
    """Shared behavior for API chat clients."""

    def __init__(self, api_key: str, model: str, api_base: str = "") -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/") if api_base else ""

    async def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        raise NotImplementedError


class AnthropicClient(BaseAPIClient):
    """Anthropic Messages API client.

    Returns Ollama-compatible shape:
    {"message": {"content": "..."}}
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_BASE = "https://api.anthropic.com"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, api_base: str = "") -> None:
        super().__init__(api_key=api_key, model=model, api_base=api_base or self.DEFAULT_BASE)

    async def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        url = f"{self.api_base}/v1/messages"
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": non_system_messages,
            "max_tokens": 2048,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                body = await resp.json()

        text = ""
        content_items = body.get("content", []) if isinstance(body, dict) else []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                text += str(item.get("text", ""))

        return {"message": {"content": text}}


class OpenAIClient(BaseAPIClient):
    """OpenAI chat-completions API client.

    Returns Ollama-compatible shape:
    {"message": {"content": "..."}}
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_BASE = "https://api.openai.com"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, api_base: str = "") -> None:
        super().__init__(api_key=api_key, model=model, api_base=api_base or self.DEFAULT_BASE)

    async def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        url = f"{self.api_base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                resp.raise_for_status()
                body = await resp.json()

        text: Optional[str] = None
        if isinstance(body, dict):
            choices = body.get("choices", [])
            if choices and isinstance(choices[0], dict):
                text = choices[0].get("message", {}).get("content")

        return {"message": {"content": text or ""}}
