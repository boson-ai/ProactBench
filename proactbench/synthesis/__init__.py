"""Scenario, blueprint, and blueprint-validation synthesis.

Each stage is a thin CLI wrapper around ``run_prompts_parallel`` that takes a
batch of ``PromptPair`` objects and returns a matching list of parsed Pydantic
responses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Type, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from ..clients import OpenAIClient, GeminiClient, AnthropicClient
from ..types import PromptPair

T = TypeVar("T", bound=BaseModel)


def _run_one(
    client,
    prompt: PromptPair,
    model: str,
    response_format: Type[T],
    gen_config: dict,
) -> Optional[T]:
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def _call():
        result = client.chat_structured(
            model=model,
            messages=[{"role": "user", "content": prompt.user}],
            system=prompt.system,
            response_format=response_format,
            **gen_config,
        )
        # chat_structured returns either the parsed obj or a (obj, usage) tuple
        return result[0] if isinstance(result, tuple) else result

    try:
        return _call()
    except Exception as e:
        print(f"[synthesis] giving up after retries: {e}", flush=True)
        return None


def run_prompts_parallel(
    client,
    prompts: list[PromptPair],
    model: str,
    response_format: Type[T],
    gen_config: dict,
    num_threads: int = 8,
    show_progress: bool = True,
    desc: str = "synthesis",
) -> list[Optional[T]]:
    """Run a batch of PromptPairs concurrently. Failed items become None."""
    def _safe(p):
        try:
            return _run_one(client, p, model, response_format, gen_config)
        except Exception as e:
            print(f"[synthesis] error: {e}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(tqdm(
            pool.map(_safe, prompts),
            total=len(prompts),
            desc=desc,
            disable=not show_progress,
        ))
