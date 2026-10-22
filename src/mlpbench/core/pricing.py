"""List prices, USD per million tokens, for the models the cost cap can meter.

Sources:
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (standard tier, global
  routing).
- OpenAI: https://developers.openai.com/api/docs/pricing (standard tier). A request whose
  prompt exceeds `long_after` input tokens is billed at the long-context rates as a whole.

A model missing here cannot be cost-capped; the CLI refuses `--max-cost` for it.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Price", "PRICES", "price_for"]


@dataclass(frozen=True)
class Price:
    input: float                  # uncached input
    output: float                 # output, reasoning/thinking included
    cache_read: float
    cache_write: float            # OpenAI cache writes; Anthropic 5-minute cache writes
    cache_write_1h: float | None = None   # Anthropic 1-hour cache writes
    long_after: int | None = None         # OpenAI: prompt size above which `long` applies
    long: "Price | None" = None

    def cost(self, *, input: int = 0, output: int = 0, cache_read: int = 0,
             cache_write: int = 0, cache_write_1h: int = 0, prompt: int | None = None) -> float:
        """Dollars for ONE request's usage (`input` = uncached input only). Never price a sum of
        requests in one call: the long-context rate is decided per request, from `prompt` (that
        request's full prompt size) or, if not given, from the tokens passed here."""
        p = self
        if prompt is None:
            prompt = input + cache_read + cache_write + cache_write_1h
        if self.long and self.long_after and prompt > self.long_after:
            p = self.long
        w1h = p.cache_write_1h if p.cache_write_1h is not None else p.cache_write
        return (input * p.input + output * p.output + cache_read * p.cache_read
                + cache_write * p.cache_write + cache_write_1h * w1h) / 1e6


def _anthropic(i: float, w5: float, w1: float, r: float, o: float) -> Price:
    return Price(input=i, output=o, cache_read=r, cache_write=w5, cache_write_1h=w1)


def _openai(i: float, r: float, w: float, o: float, li: float, lr: float, lw: float,
            lo: float) -> Price:
    return Price(input=i, output=o, cache_read=r, cache_write=w, long_after=272_000,
                 long=Price(input=li, output=lo, cache_read=lr, cache_write=lw))


PRICES: dict[str, Price] = {
    # Anthropic: base input, 5m write, 1h write, cache hit, output
    "claude-opus-5": _anthropic(5, 6.25, 10, 0.50, 25),
    "claude-opus-4-8": _anthropic(5, 6.25, 10, 0.50, 25),
    "claude-sonnet-5": _anthropic(2, 2.50, 4, 0.20, 10),
    "claude-sonnet-4-6": _anthropic(3, 3.75, 6, 0.30, 15),
    # OpenAI, GPT-5.6 and later: input, cached input, cache write, output; then the same
    # four above 272K prompt tokens
    "gpt-5.6-sol": _openai(4, 0.40, 5, 20, 8, 0.80, 10, 30),
    "gpt-5.6-terra": _openai(2, 0.20, 2.5, 12, 4, 0.40, 5, 18),
    "gpt-5.6-luna": _openai(0.20, 0.02, 0.25, 1.20, 0.40, 0.04, 0.50, 1.80),
    "gpt-6-astra": _openai(10, 1, 12.5, 50, 20, 2, 25, 75),
    "gpt-6-sol": _openai(2, 0.20, 2.5, 10, 4, 0.40, 5, 15),
    "gpt-6-luna": _openai(0.10, 0.01, 0.125, 0.50, 0.20, 0.02, 0.25, 0.75),
    "gpt-6.1-sol": _openai(2, 0.10, 2.5, 10, 4, 0.20, 5, 15),
}


def price_for(model_id: str) -> Price | None:
    return PRICES.get(model_id)
