"""Minimal API clients for ProactBench.

Each client implements two methods that the rest of the benchmark depends on:

  * ``chat(model, messages, system=None, **gen)`` — returns the assistant's
    text response (and optional usage dict). This is the unstructured
    multi-turn chat path used by Planner/User-Agent role-play and by the
    evaluated model in online/offline eval.

  * ``chat_structured(model, messages, response_format, system=None, **gen)`` —
    returns a parsed Pydantic object matching ``response_format``.
    Used by the judge in the evaluation loop.

Supported providers:

  * OpenAI (``gpt-*``, ``o1*``, ``o3*``, ``o4*``)
  * Gemini (``gemini-*``)
  * Anthropic (``claude-*``), structured via tool-use
  * OpenAI-compatible endpoints (set ``base_url=...``)

The clients are deliberately thin — they translate a common gen_config into
provider-specific kwargs and handle the two ways each provider exposes
structured output. Audio, vision, and streaming are out of scope.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Type, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

T = TypeVar("T", bound=BaseModel)


# ── Shared gen-config shape ───────────────────────────────────────────────────

@dataclass
class GenConfig:
    """Provider-agnostic generation knobs.

    The clients translate these into provider-specific fields. Fields that a
    given provider doesn't support are silently ignored.
    """
    temperature: float = 0.7
    max_new_tokens: int = 8192
    top_p: Optional[float] = None
    # OpenAI reasoning models (gpt-5, o1, o3, o4)
    reasoning_effort: Optional[str] = None  # "minimal" | "low" | "medium" | "high"
    # Claude / Gemini extended thinking
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── OpenAI (and OpenAI-compatible endpoints) ──────────────────────────────────

_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_openai_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _OPENAI_REASONING_PREFIXES)


def _is_validation_or_schema_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("ValidationError", "LengthFinishReasonError", "ContentFilterFinishReasonError"):
        return True
    msg = str(exc).lower()
    return "validation error" in msg or "should be a valid" in msg or "model_validate" in msg


def _coerce_strings_to_dicts(obj):
    """Recursively replace string values that look like JSON objects with the
    decoded dict. Some OpenAI-compatible providers (e.g. Kimi via OpenRouter)
    return nested fields as JSON-encoded strings instead of nested objects."""
    import json as _json
    if isinstance(obj, dict):
        return {k: _coerce_strings_to_dicts(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_strings_to_dicts(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return _coerce_strings_to_dicts(_json.loads(s))
            except _json.JSONDecodeError:
                return obj
    return obj


def _tolerant_pydantic_parse(text: str, response_format: Type[BaseModel]):
    import json as _json
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    raw = _json.loads(s)
    fixed = _coerce_strings_to_dicts(raw)
    return response_format.model_validate(fixed)


class OpenAIClient:
    """Wrapper around the OpenAI Python SDK (also works with any
    OpenAI-compatible endpoint via ``base_url``)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 300.0):
        import openai
        self._openai = openai
        key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout)

    # ----------------------------- helpers -----------------------------------

    def _build_kwargs(self, model: str, gen: dict) -> dict:
        kwargs: dict = {}
        if _is_openai_reasoning(model):
            kwargs["max_completion_tokens"] = gen.get("max_new_tokens", 8192)
            if gen.get("reasoning_effort"):
                kwargs["reasoning_effort"] = gen["reasoning_effort"]
            # Reasoning models require temperature=1.0 and don't accept top_p.
            kwargs["temperature"] = 1.0
        else:
            kwargs["max_tokens"] = gen.get("max_new_tokens", 8192)
            kwargs["temperature"] = gen.get("temperature", 0.7)
            if gen.get("top_p") is not None:
                kwargs["top_p"] = gen["top_p"]
        return kwargs

    @staticmethod
    def _usage(resp) -> Optional[dict]:
        if resp.usage is None:
            return None
        return {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    @staticmethod
    def _format_messages(messages: list[dict], system: Optional[str]) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            out.append({"role": m["role"], "content": m["content"]})
        return out

    # ----------------------------- public API --------------------------------

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat(self, model: str, messages: list[dict], system: Optional[str] = None,
             return_usage: bool = False, **gen):
        kwargs = self._build_kwargs(model, gen)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=self._format_messages(messages, system),
                **kwargs,
            )
            text = resp.choices[0].message.content
            return (text, self._usage(resp)) if return_usage else text
        except Exception as e:
            print(f"[OpenAIClient.chat] retry after error: {e}", flush=True)
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat_structured(self, model: str, messages: list[dict],
                        response_format: Type[T], system: Optional[str] = None,
                        return_usage: bool = False, **gen) -> Any:
        kwargs = self._build_kwargs(model, gen)
        formatted = self._format_messages(messages, system)
        try:
            resp = self.client.beta.chat.completions.parse(
                model=model,
                messages=formatted,
                response_format=response_format,
                **kwargs,
            )
            parsed = resp.choices[0].message.parsed
            return (parsed, self._usage(resp)) if return_usage else parsed
        except Exception as parse_err:
            if not _is_validation_or_schema_error(parse_err):
                print(f"[OpenAIClient.chat_structured] retry after error: {parse_err}", flush=True)
                raise
            # Fallback for OpenAI-compatible providers (Kimi/DeepSeek via OpenRouter)
            # that return nested fields as JSON-encoded strings instead of dicts.
            print(f"[OpenAIClient.chat_structured] strict parse failed, "
                  f"falling back to json_object: {parse_err}", flush=True)
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=formatted,
                    response_format={"type": "json_object"},
                    **kwargs,
                )
            except Exception as e:
                print(f"[OpenAIClient.chat_structured] fallback failed: {e}", flush=True)
                raise
            parsed = _tolerant_pydantic_parse(resp.choices[0].message.content,
                                              response_format)
            return (parsed, self._usage(resp)) if return_usage else parsed


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiClient:
    """Wrapper around google-genai for Gemini models."""

    def __init__(self, api_key: Optional[str] = None, thinking_budget: Optional[int] = None):
        from google import genai
        from google.genai import types as genai_types
        self._genai = genai
        self._types = genai_types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)
        self._default_thinking_budget = thinking_budget
        self._safety_settings = [
            genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in ("HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
        ]

    def _thinking_config(self, budget: Optional[int]):
        if budget is None:
            return None
        return self._types.ThinkingConfig(
            include_thoughts=(budget > 0),
            thinking_budget=budget,
        )

    def _gen_config(self, gen: dict, system: Optional[str], response_schema=None):
        budget = gen.get("thinking_budget", self._default_thinking_budget)
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=gen.get("max_new_tokens", 8192),
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p"),
            safety_settings=self._safety_settings,
            thinking_config=self._thinking_config(budget),
            response_mime_type="application/json" if response_schema else None,
            response_json_schema=response_schema,
        )
        return cfg

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat(self, model: str, messages: list[dict], system: Optional[str] = None,
             return_usage: bool = False, **gen):
        contents = [m["content"] for m in messages]
        try:
            resp = self.client.models.generate_content(
                model=model, contents=contents,
                config=self._gen_config(gen, system),
            )
            text = resp.text
            return (text, None) if return_usage else text
        except Exception as e:
            print(f"[GeminiClient.chat] retry after error: {e}", flush=True)
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat_structured(self, model: str, messages: list[dict],
                        response_format: Type[T], system: Optional[str] = None,
                        return_usage: bool = False, **gen) -> Any:
        contents = [m["content"] for m in messages]
        try:
            resp = self.client.models.generate_content(
                model=model, contents=contents,
                config=self._gen_config(gen, system, response_schema=response_format.model_json_schema()),
            )
            parsed = response_format.model_validate_json(resp.text)
            return (parsed, None) if return_usage else parsed
        except Exception as e:
            print(f"[GeminiClient.chat_structured] retry after error: {e}", flush=True)
            raise


# ── Anthropic / Claude ────────────────────────────────────────────────────────

_CLAUDE_THINKING_PREFIXES = (
    "claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-3-7-sonnet",
)

# Claude models that have deprecated all sampling parameters
# (``temperature``, ``top_p``, ``top_k``). Passing any of them returns 400.
# Affects Claude Opus/Sonnet/Haiku 4.7+.
_CLAUDE_NO_SAMPLING_PARAMS_PREFIXES = (
    "claude-opus-4-7", "claude-sonnet-4-7", "claude-haiku-4-7",
)


def _claude_supports_thinking(model: str) -> bool:
    return any(model.startswith(p) for p in _CLAUDE_THINKING_PREFIXES)


def _claude_accepts_sampling_params(model: str) -> bool:
    """Returns False for models that deprecated temperature/top_p/top_k."""
    return not any(model.startswith(p) for p in _CLAUDE_NO_SAMPLING_PARAMS_PREFIXES)


class AnthropicClient:
    """Wrapper around the Anthropic SDK.

    Structured output is produced via Anthropic tool-use (the response_format
    Pydantic schema is exposed as a single tool)."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 600.0):
        import anthropic
        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout)

    def _build_kwargs(self, model: str, gen: dict) -> dict:
        max_new = gen.get("max_new_tokens", 8192)
        kwargs: dict = {"max_tokens": max_new}
        accepts_sampling = _claude_accepts_sampling_params(model)
        if accepts_sampling:
            kwargs["temperature"] = gen.get("temperature", 0.7)
            top_p = gen.get("top_p")
            if top_p is not None:
                kwargs["top_p"] = top_p
            top_k = gen.get("top_k")
            if top_k is not None:
                kwargs["top_k"] = top_k
        if gen.get("enable_thinking") and _claude_supports_thinking(model):
            budget = gen.get("thinking_budget") or max(max_new // 2, 1024)
            budget = min(budget, max(max_new - 1, 1))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Extended thinking requires temperature=1.0 and disallows top_p/top_k.
            # Claude 4.7+ deprecated all sampling params; for those, pass none.
            if accepts_sampling:
                kwargs["temperature"] = 1.0
                kwargs.pop("top_p", None)
                kwargs.pop("top_k", None)
        return kwargs

    @staticmethod
    def _extract_text(resp) -> str:
        return "".join(b.text for b in (resp.content or []) if getattr(b, "type", None) == "text")

    @staticmethod
    def _usage(resp) -> Optional[dict]:
        u = getattr(resp, "usage", None)
        if u is None:
            return None
        p = getattr(u, "input_tokens", 0) or 0
        c = getattr(u, "output_tokens", 0) or 0
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat(self, model: str, messages: list[dict], system: Optional[str] = None,
             return_usage: bool = False, **gen):
        kwargs = self._build_kwargs(model, gen)
        if system:
            kwargs["system"] = system
        try:
            resp = self.client.messages.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                **kwargs,
            )
            text = self._extract_text(resp)
            return (text, self._usage(resp)) if return_usage else text
        except Exception as e:
            print(f"[AnthropicClient.chat] retry after error: {e}", flush=True)
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    def chat_structured(self, model: str, messages: list[dict],
                        response_format: Type[T], system: Optional[str] = None,
                        return_usage: bool = False, **gen) -> Any:
        schema = response_format.model_json_schema()
        tool_name = "structured_output"
        tool = {
            "name": tool_name,
            "description": (
                f"Return the answer by calling this tool exactly once with an "
                f"argument object matching the {response_format.__name__} schema."
            ),
            "input_schema": schema,
        }
        kwargs = self._build_kwargs(model, gen)
        if system:
            kwargs["system"] = system
        kwargs["tools"] = [tool]
        if "thinking" not in kwargs:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        try:
            resp = self.client.messages.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                **kwargs,
            )
            parsed = None
            for block in (resp.content or []):
                if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                    parsed = response_format.model_validate(block.input)
                    break
            if parsed is None:
                # Fallback: plain-text JSON (rare; happens under extended thinking).
                text = self._extract_text(resp).strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:]
                    text = text.strip()
                parsed = response_format.model_validate_json(text)
            return (parsed, self._usage(resp)) if return_usage else parsed
        except Exception as e:
            print(f"[AnthropicClient.chat_structured] retry after error: {e}", flush=True)
            raise


# ── Factory ───────────────────────────────────────────────────────────────────

# Third-party OpenAI-compatible providers. Map model-name prefix → base URL +
# env var holding the API key. Add a new provider here once and ``make_client``
# picks it up automatically.
_OAI_COMPAT_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("kimi",     "https://api.moonshot.ai/v1",   "MOONSHOT_API_KEY"),
    ("moonshot", "https://api.moonshot.ai/v1",   "MOONSHOT_API_KEY"),
    ("deepseek", "https://api.deepseek.com/v1",  "DEEPSEEK_API_KEY"),
)

# Map well-known OpenAI-compatible base URLs → env var holding the key.
# Used when the caller passes ``base_url=...`` directly (e.g. OpenRouter).
_BASE_URL_KEY_VARS: dict[str, str] = {
    "openrouter.ai": "OPENROUTER_API_KEY",
    "api.moonshot.ai": "MOONSHOT_API_KEY",
    "api.deepseek.com": "DEEPSEEK_API_KEY",
    "api.together.xyz": "TOGETHER_API_KEY",
    "api.fireworks.ai": "FIREWORKS_API_KEY",
}


def _api_key_for_base_url(base_url: str) -> str:
    """Pick the right env var for a given base URL. Falls back to OPENAI_API_KEY."""
    for host, key_var in _BASE_URL_KEY_VARS.items():
        if host in base_url:
            return os.environ.get(key_var, "EMPTY")
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def make_client(model: str, base_url: Optional[str] = None, **kwargs):
    """Return the right client for ``model``.

    Resolution rules:

      * ``base_url`` provided → OpenAI-compatible endpoint
      * model starts with ``claude-`` → AnthropicClient
      * model contains ``gemini`` → GeminiClient
      * model matches a known third-party prefix (``kimi-``, ``deepseek-``, …) →
        OpenAIClient with that provider's base URL and key
      * everything else → OpenAIClient (works for ``gpt-*``, ``o1*`` … and any
        OpenAI-compatible model if the env has OPENAI_API_KEY)
    """
    if base_url is not None:
        return OpenAIClient(api_key=_api_key_for_base_url(base_url), base_url=base_url)
    if model.startswith("claude-"):
        return AnthropicClient(**kwargs)
    if "gemini" in model:
        return GeminiClient(**kwargs)
    for prefix, provider_url, key_var in _OAI_COMPAT_PROVIDERS:
        if model.startswith(prefix):
            return OpenAIClient(api_key=os.environ.get(key_var, "EMPTY"), base_url=provider_url)
    return OpenAIClient(**kwargs)
