"""Token pricing for the AdaL cloud proxy, calibrated against real billing.

Every rate below was MEASURED this session by differencing
``GET https://adal.sylph.ai/api/credits/balance`` around isolated single calls
(USD per 1M tokens). The observed rates already include each model's promotion
(luna 80 % off, terra/sol 20 % off), so they are used verbatim from
:class:`RATES` and are NEVER re-derived from catalog ``meta_display``
multipliers.

Two billing conventions exist upstream and both were measured:

- Anthropic: ``input_tokens`` EXCLUDES cached and newly-written tokens, so all
  four token kinds bill at their own rate.
- OpenAI (chat + responses): ``prompt_tokens`` INCLUDES cached tokens, so the
  uncached portion bills at ``input`` and the cached portion at ``cache_read``.
"""

from __future__ import annotations

from dataclasses import dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class Rate:
    """USD per 1M tokens for one upstream model."""

    input: float
    output: float
    cache_write: float
    cache_read: float


_SONNET = Rate(input=3.00, output=15.00, cache_write=3.75, cache_read=0.30)
_OPUS = Rate(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50)
_FABLE = Rate(input=7.50, output=37.50, cache_write=9.375, cache_read=0.75)
_TERRA = Rate(input=2.00, output=12.00, cache_write=2.00, cache_read=0.20)
_LUNA = Rate(input=0.20, output=1.20, cache_write=0.20, cache_read=0.02)
_SOL = Rate(input=4.00, output=24.00, cache_write=4.00, cache_read=0.40)

#: Calibrated rates keyed by upstream ``model_id``.
RATES: dict[str, Rate] = {
    "claude-sonnet-4-6": _SONNET,
    "claude-sonnet-5": _SONNET,
    "claude-opus-4-6": _OPUS,
    "claude-opus-5": _OPUS,
    "claude-fable-5-1": _FABLE,
    "gpt-5.6-terra": _TERRA,
    "gpt-5.6-luna": _LUNA,
    "gpt-5.6-sol": _SOL,
}

#: Mid-tier fallback row per catalog provider, used when no exact rate matches.
PROVIDER_DEFAULT_RATE: dict[str, Rate] = {
    "anthropic": _SONNET,
    "zai": _SONNET,
    "minimax": _SONNET,
    "xai": _SONNET,
    "openai": _TERRA,
    "chatgpt_web": _TERRA,
    "deepseek": _TERRA,
    "kimi": _TERRA,
    "qwen": _TERRA,
    "meta": _TERRA,
}

# Longest first so a longer provider key always wins prefix matching.
_PROVIDER_PREFIXES: tuple[str, ...] = tuple(
    sorted(PROVIDER_DEFAULT_RATE, key=len, reverse=True)
)


def rate_for(model: str, provider: str = "") -> tuple[Rate, bool]:
    """Resolve the billing rate for *model*.

    Returns ``(rate, estimated)``: exact :data:`RATES` hits (including after
    stripping a leading catalog-provider prefix such as ``chatgpt_web-``) are
    not estimated; provider mid-tier fallbacks are.
    """
    rate = RATES.get(model)
    if rate is not None:
        return rate, False
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix + "-"):
            rate = RATES.get(model[len(prefix) + 1 :])
            if rate is not None:
                return rate, False
    rate = PROVIDER_DEFAULT_RATE.get(provider)
    if rate is not None:
        return rate, True
    return _SONNET, True


def cost_for(
    *,
    model: str,
    provider: str,
    protocol: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> tuple[float, bool]:
    """Cost in USD for one request, rounded to 6 decimal places.

    *protocol* selects the measured billing convention: the Anthropic
    convention (``input_tokens`` excludes cached/newly-written tokens) for
    ``"anthropic"``, the OpenAI convention (``prompt_tokens`` includes cached
    tokens) for ``"openai_chat"``/``"responses"``. Any unknown protocol uses
    the Anthropic convention.
    """
    rate, estimated = rate_for(model, provider)
    if protocol in ("openai_chat", "responses"):
        uncached = max(input_tokens - cache_read_tokens, 0)
        micro = (
            uncached * rate.input
            + cache_read_tokens * rate.cache_read
            + output_tokens * rate.output
        )
    else:
        micro = (
            input_tokens * rate.input
            + output_tokens * rate.output
            + cache_write_tokens * rate.cache_write
            + cache_read_tokens * rate.cache_read
        )
    return round(micro / _PER_MILLION, 6), estimated
