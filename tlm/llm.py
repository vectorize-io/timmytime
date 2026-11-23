"""Tiny client for the OpenAI-style chat endpoint (hindsight dev inference, Gemma).

Reads TLM_LLM_URL, TLM_LLM_MODEL and TLM_LLM_KEY (falls back to
HINDSIGHT_INFERENCE_DEV_API_KEY). Stdlib only: this is dev tooling.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("TLM_LLM_URL", "https://inference.dev.hindsight.vectorize.io/v1/chat/completions")
MODEL = os.environ.get("TLM_LLM_MODEL", "gemma-4-26b-a4b")


def chat(prompt: str, system: str | None = None, temperature: float = 1.0, max_tokens: int = 1500) -> str:
    key = os.environ.get("TLM_LLM_KEY") or os.environ["HINDSIGHT_INFERENCE_DEV_API_KEY"]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
    for attempt in range(6):
        req = urllib.request.Request(URL, body.encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ConnectionError) as e:
            if attempt == 5:
                raise
            time.sleep(2**attempt + (5 if getattr(e, "code", 0) == 429 else 0))
    raise AssertionError("unreachable")


def json_list(text: str) -> list:
    """The first JSON array in an LLM reply, or [] if there is none."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        out = json.loads(m[0])
    except json.JSONDecodeError:
        return []
    return out if isinstance(out, list) else []


def run_all(fn, items, workers: int = 32):
    """fn over items in parallel; a failed item gives None instead of stopping the batch."""
    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # one bad call must not lose the whole batch
            print(f"llm call failed: {type(e).__name__}: {e}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(safe, items))
