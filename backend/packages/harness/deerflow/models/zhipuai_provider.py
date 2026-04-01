"""ZhipuAI ChatOpenAI provider with retry logic and exponential backoff.

Handles rate limiting (429 errors) gracefully by retrying with exponential backoff.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_BACKOFF_MS = 2000
MAX_BACKOFF_MS = 60000


class ZhipuAIChatModel(ChatOpenAI):
    """ChatOpenAI with retry logic and exponential backoff for ZhipuAI.

    Handles rate limiting (429 errors) by retrying with exponential backoff.

    Config example:
        - name: glm-5
          display_name: GLM-5
          use: deerflow.models.zhipuai_provider:ZhipuAIChatModel
          model: glm-5
          openai_api_key: $ZHIPUAI_API_KEY
          openai_api_base: https://open.bigmodel.cn/api/paas/v4/
          max_tokens: 4096
          temperature: 0.3
          retry_max_attempts: 5
    """

    retry_max_attempts: int = MAX_RETRIES

    def _validate_retry_config(self) -> None:
        if self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be >= 1")

    def model_post_init(self, __context: Any) -> None:
        self._validate_retry_config()
        super().model_post_init(__context)

    def _calc_backoff_ms(self, attempt: int) -> int:
        """Exponential backoff with jitter."""
        backoff_ms = BASE_BACKOFF_MS * (2 ** (attempt - 1))
        backoff_ms = min(backoff_ms, MAX_BACKOFF_MS)
        jitter_ms = random.randint(0, int(backoff_ms * 0.2))
        return backoff_ms + jitter_ms

    def _should_retry(self, status_code: int) -> bool:
        """Check if we should retry based on status code."""
        return status_code in (429, 500, 502, 503, 504)

    def invoke(
        self,
        input: LanguageModelInput,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Override with retry logic."""
        last_exception = None

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return super().invoke(input, config=config, **kwargs)
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code is None and hasattr(e, "response"):
                    status_code = getattr(e.response, "status_code", None)

                if status_code and self._should_retry(status_code):
                    if attempt >= self.retry_max_attempts:
                        raise
                    wait_ms = self._calc_backoff_ms(attempt)
                    logger.warning(
                        f"ZhipuAI API error {status_code}, retrying {attempt}/{self.retry_max_attempts} after {wait_ms}ms"
                    )
                    time.sleep(wait_ms / 1000)
                    last_exception = e
                else:
                    raise

        raise last_exception

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Override with retry logic (async)."""
        import asyncio

        last_exception = None

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return await super().ainvoke(input, config=config, **kwargs)
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code is None and hasattr(e, "response"):
                    status_code = getattr(e.response, "status_code", None)

                if status_code and self._should_retry(status_code):
                    if attempt >= self.retry_max_attempts:
                        raise
                    wait_ms = self._calc_backoff_ms(attempt)
                    logger.warning(
                        f"ZhipuAI API error {status_code}, retrying {attempt}/{self.retry_max_attempts} after {wait_ms}ms"
                    )
                    await asyncio.sleep(wait_ms / 1000)
                    last_exception = e
                else:
                    raise

        raise last_exception
