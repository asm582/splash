"""Measures the cost of re-rendering an image request's chat template.

transformers caches compiled Jinja templates keyed by the template *source
string* (utils.chat_template_utils._compile_jinja_template). Frontend's
image path used to substitute a fresh random marker into that source on
every request, which made the cache string differ every time and forced a
full Jinja recompile per image request that a text-only request never pays.

Run as ``python -m dev.benchmarks.template_cache_bench --tokenizer PATH``,
where PATH is a directory produced by ``make install`` (for example
``install/models/incoai/Qwen3.8-27B-Splash/tokenizer``); no model weights or
native engine are needed since only the tokenizer and chat template load.
"""

from __future__ import annotations

import argparse
import secrets
import statistics
import time

from transformers import AutoTokenizer
from transformers.utils import chat_template_utils

from server.api_shapes import IMAGE_PAD_TOKEN
from server.frontend import IMAGE_RENDER_MARKER


def build_messages(prior_turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "You are a careful coding agent."}]
    for turn in range(prior_turns):
        messages.append(
            {
                "role": "user",
                "content": f"Turn {turn}: look at file_{turn}.py and summarize the diff.\n"
                + ("line of code\n" * 20),
            }
        )
        messages.append(
            {"role": "assistant", "content": f"Summary for turn {turn}: looks fine."}
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what does this screenshot show?"},
                {"type": "image_url", "image_url": {"url": "unused"}},
            ],
        }
    )
    return messages


def render_once(tokenizer, messages, template, marker: str) -> str:
    source = tokenizer.get_chat_template(tools=template.get("tools"))
    return tokenizer.apply_chat_template(
        messages,
        **{
            **template,
            "tokenize": False,
            "chat_template": source.replace(IMAGE_PAD_TOKEN, marker),
        },
    )


def bench(tokenizer, messages, template, *, random_marker: bool, iterations: int):
    chat_template_utils._compile_jinja_template.cache_clear()
    times = []
    for _ in range(iterations):
        marker = (
            f"__splash_image_{secrets.token_hex(16)}__"
            if random_marker
            else IMAGE_RENDER_MARKER
        )
        started = time.perf_counter()
        render_once(tokenizer, messages, template, marker)
        times.append(time.perf_counter() - started)
    return times, chat_template_utils._compile_jinja_template.cache_info()


def report(label, times, info) -> None:
    print(
        f"  {label}: mean={statistics.mean(times) * 1000:.3f}ms "
        f"median={statistics.median(times) * 1000:.3f}ms "
        f"(jinja cache hits={info.hits} misses={info.misses})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--turns", default="1,5,20")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    template = {"tokenize": False, "return_dict": False, "add_generation_prompt": True}

    for prior_turns in (int(value) for value in args.turns.split(",")):
        messages = build_messages(prior_turns)
        print(f"conversation with {prior_turns} prior turns + 1 image turn:")
        before, before_info = bench(
            tokenizer,
            messages,
            template,
            random_marker=True,
            iterations=args.iterations,
        )
        report("fresh random marker per request (before)", before, before_info)
        after, after_info = bench(
            tokenizer,
            messages,
            template,
            random_marker=False,
            iterations=args.iterations,
        )
        report("fixed process-lifetime marker (after)", after, after_info)
        speedup = statistics.mean(before) / statistics.mean(after)
        print(f"  => {speedup:.1f}x faster mean template render time\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
