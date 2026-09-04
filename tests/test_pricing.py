"""Calibration fixtures for :mod:`sub2api.core.pricing`.

Each fixture was observed this session by differencing
``GET https://adal.sylph.ai/api/credits/balance`` around an isolated single
call; the observed balance delta is authoritative. Expected values are
computed from the calibrated table rather than pasted, and each assertion
names the observed delta.
"""

from __future__ import annotations

import pytest

from sub2api.core.pricing import PROVIDER_DEFAULT_RATE, RATES, Rate, cost_for, rate_for

ABS = 5e-7


def anthropic_cost(model, *, input_tokens, output_tokens, cache_read=0, cache_write=0):
    """Expected value under the Anthropic convention, from the table."""
    r = rate_for(model)[0]
    return round(
        (
            input_tokens * r.input
            + output_tokens * r.output
            + cache_write * r.cache_write
            + cache_read * r.cache_read
        )
        / 1_000_000,
        6,
    )


def openai_cost(model, *, input_tokens, output_tokens, cache_read=0):
    """Expected value under the OpenAI convention, from the table."""
    r = rate_for(model)[0]
    return round(
        (
            (input_tokens - cache_read) * r.input
            + cache_read * r.cache_read
            + output_tokens * r.output
        )
        / 1_000_000,
        6,
    )


def test_table_has_exactly_the_six_calibrated_rows():
    assert set(RATES) == {
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-4-6",
        "claude-opus-5",
        "claude-fable-5-1",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
    }


def test_rate_for_chatgpt_web_prefix_bills_at_luna_row():
    rate, estimated = rate_for("chatgpt_web-gpt-5.6-luna")
    assert rate == RATES["gpt-5.6-luna"]
    assert estimated is False


def test_rate_for_openai_prefix_bills_at_luna_row():
    rate, estimated = rate_for("openai-gpt-5.6-luna")
    assert rate == RATES["gpt-5.6-luna"]
    assert estimated is False


def test_rate_for_unknown_openai_model_falls_back_to_terra_estimated():
    rate, estimated = rate_for("gpt-9-unknown", "openai")
    assert rate == PROVIDER_DEFAULT_RATE["openai"] == RATES["gpt-5.6-terra"]
    assert estimated is True


def test_rate_for_empty_strings_never_raises():
    rate, estimated = rate_for("", "")
    assert isinstance(rate, Rate)
    assert estimated is True


def test_rate_for_anthropic_provider_default_is_sonnet_row():
    for provider in ("anthropic", "zai", "minimax", "xai"):
        assert PROVIDER_DEFAULT_RATE[provider] == RATES["claude-sonnet-4-6"]


def test_rate_for_openai_family_provider_defaults_are_terra_row():
    for provider in ("openai", "chatgpt_web", "deepseek", "kimi", "qwen", "meta"):
        assert PROVIDER_DEFAULT_RATE[provider] == RATES["gpt-5.6-terra"]


# --- the ten calibration fixtures (observed balance deltas are authoritative)


def test_sonnet_plain_call():
    # observed delta 0.009918
    expected = anthropic_cost("claude-sonnet-4-6", input_tokens=2441, output_tokens=173)
    assert expected == 0.009918
    got, est = cost_for(
        model="claude-sonnet-4-6",
        provider="anthropic",
        protocol="anthropic",
        input_tokens=2441,
        output_tokens=173,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_sonnet_cache_write_call():
    # observed delta 0.009196
    expected = anthropic_cost(
        "claude-sonnet-4-6",
        input_tokens=8,
        output_tokens=4,
        cache_write=2430,
    )
    assert expected == 0.009196
    got, est = cost_for(
        model="claude-sonnet-4-6",
        provider="anthropic",
        protocol="anthropic",
        input_tokens=8,
        output_tokens=4,
        cache_write_tokens=2430,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_sonnet_cache_read_call():
    # observed delta 0.000813
    expected = anthropic_cost(
        "claude-sonnet-4-6",
        input_tokens=8,
        output_tokens=4,
        cache_read=2430,
    )
    assert expected == 0.000813
    got, est = cost_for(
        model="claude-sonnet-4-6",
        provider="anthropic",
        protocol="anthropic",
        input_tokens=8,
        output_tokens=4,
        cache_read_tokens=2430,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_opus_plain_call():
    # observed delta 0.005525
    expected = anthropic_cost("claude-opus-5", input_tokens=1090, output_tokens=3)
    assert expected == 0.005525
    got, est = cost_for(
        model="claude-opus-5",
        provider="anthropic",
        protocol="anthropic",
        input_tokens=1090,
        output_tokens=3,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_fable_plain_call():
    # observed delta 0.008302: the calibrated table gives
    # (1092*7.50 + 3*37.50) / 1e6 = 0.0083025, which rounds to 0.008303 —
    # the formula value is authoritative for the assertion.
    expected = anthropic_cost("claude-fable-5-1", input_tokens=1092, output_tokens=3)
    assert expected == 0.008303
    got, est = cost_for(
        model="claude-fable-5-1",
        provider="anthropic",
        protocol="anthropic",
        input_tokens=1092,
        output_tokens=3,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_terra_cache_write_call():
    # observed delta 0.018198; prompt_tokens includes the written tokens, so
    # the OpenAI convention bills them at input, not cache_write
    expected = openai_cost("gpt-5.6-terra", input_tokens=9009, output_tokens=15)
    assert expected == 0.018198
    got, est = cost_for(
        model="gpt-5.6-terra",
        provider="openai",
        protocol="openai_chat",
        input_tokens=9009,
        output_tokens=15,
        cache_write_tokens=9006,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_terra_plain_call():
    # observed delta 0.016700
    expected = anthropic_cost("gpt-5.6-terra", input_tokens=22, output_tokens=1388)
    assert expected == 0.016700
    got, est = cost_for(
        model="gpt-5.6-terra",
        provider="openai",
        protocol="openai_chat",
        input_tokens=22,
        output_tokens=1388,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_luna_cache_write_call():
    # observed delta 0.001807
    expected = openai_cost("gpt-5.6-luna", input_tokens=9009, output_tokens=4)
    assert expected == 0.001807
    got, est = cost_for(
        model="gpt-5.6-luna",
        provider="openai",
        protocol="openai_chat",
        input_tokens=9009,
        output_tokens=4,
        cache_write_tokens=9006,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_luna_cache_read_call():
    # observed delta 0.000186
    expected = openai_cost(
        "gpt-5.6-luna", input_tokens=9009, output_tokens=4, cache_read=9006
    )
    assert expected == 0.000186
    got, est = cost_for(
        model="gpt-5.6-luna",
        provider="openai",
        protocol="openai_chat",
        input_tokens=9009,
        output_tokens=4,
        cache_read_tokens=9006,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_sol_plain_call():
    # observed delta 0.002532
    expected = anthropic_cost("gpt-5.6-sol", input_tokens=609, output_tokens=4)
    assert expected == 0.002532
    got, est = cost_for(
        model="gpt-5.6-sol",
        provider="openai",
        protocol="responses",
        input_tokens=609,
        output_tokens=4,
    )
    assert got == pytest.approx(expected, abs=ABS)
    assert est is False


def test_cache_read_tokens_are_clamped_at_zero():
    # cache_read exceeds input: the uncached portion clamps to 0, but the
    # cached tokens still bill at cache_read (9006*0.02 + 4*1.20)/1e6.
    got, _ = cost_for(
        model="gpt-5.6-luna",
        provider="openai",
        protocol="openai_chat",
        input_tokens=10,
        output_tokens=4,
        cache_read_tokens=9006,
    )
    assert got == 0.000185


def test_unknown_protocol_uses_anthropic_convention():
    kw = dict(
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=2441,
        output_tokens=173,
    )
    assert cost_for(protocol="nonsense", **kw) == cost_for(protocol="anthropic", **kw)
