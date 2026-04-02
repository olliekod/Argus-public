# Created by Oliver Meihls

# Async resource coordination primitives for Argus agents.

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AgentResourceManager:
    # Coordinates bounded access to LLM backends.
    #
    # The debate protocol can trigger multiple parallel completions. Limiting
    # concurrent calls avoids over-saturating local inference services.

    def __init__(self, max_concurrent_llm_calls: int = 2) -> None:
        if max_concurrent_llm_calls < 1:
            raise ValueError("max_concurrent_llm_calls must be >= 1")
        self.max_concurrent_llm_calls = max_concurrent_llm_calls
        self._llm_semaphore = asyncio.Semaphore(max_concurrent_llm_calls)
        self.gpu_enabled = True  # Default to True for LLM coordination

    @asynccontextmanager
    async def llm_slot(self) -> AsyncIterator[None]:
        # Acquire one bounded completion slot for an LLM call.
        await self._llm_semaphore.acquire()
        try:
            yield
        finally:
            self._llm_semaphore.release()

