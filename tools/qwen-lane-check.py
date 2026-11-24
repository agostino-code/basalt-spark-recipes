#!/usr/bin/env python3
"""Acceptance check for a running Basalt lane serving Qwen3.8-Flash-Next.

The Hy4-flash lane on this box faulted in prefill when a large prompt prefix was reused and then a
small chunk was appended (README, "The two sm_121 faults"). This check exercises that shape on the
Qwen lane and fails if any request fails or the lane log records a fault:

    R1  16,522 tokens                 -> OK
    R2  33,022 tokens                 -> OK
    R3  33,390 tokens (R2 + ~350)     -> OK

Usage:
    python3 tools/qwen-lane-check.py --base http://127.0.0.1:30001 --log /path/to/lane.log
    python3 tools/qwen-lane-check.py --base ... --log ... --ladder 4096:8192:8400

Exit status: 0 when every request finished and the log is clean, 1 otherwise.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

FILLER = ("The committee reviewed the quarterly figures and noted that several regional offices had "
          "exceeded their targets, while others reported delays caused by supply chain disruptions, "
          "staffing changes and unusually severe weather. ")


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def count_tokens(base, model, text):
    try:
        return post(base, "/v1/messages/count_tokens",
                    {"model": model, "messages": [{"role": "user", "content": text}]})["input_tokens"]
    except (urllib.error.HTTPError, KeyError):
        return max(1, len(text) // 4)


def text_of(base, model, tokens):
    """Filler text that the server counts as about `tokens` prompt tokens."""
    per_char = count_tokens(base, model, FILLER) / float(len(FILLER))
    pieces = max(1, int(tokens / max(per_char * len(FILLER), 1)))
    text = FILLER * pieces
    have = count_tokens(base, model, text)
    # trim or extend by a few copies until within 1% of the target
    for _ in range(20):
        if abs(have - tokens) <= max(64, tokens // 100):
            break
        if have < tokens:
            text += FILLER * max(1, int((tokens - have) / max(per_char * len(FILLER), 1)))
        else:
            cut = int((have - tokens) / max(per_char, 1e-9))
            text = text[:-cut] if cut < len(text) else text[: max(1, len(text) // 2)]
        have = count_tokens(base, model, text)
    return text


def ask(base, model, text, max_tokens=16):
    t0 = time.perf_counter()
    r = post(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens, "temperature": 0,
    })
    wall = time.perf_counter() - t0
    usage = r.get("usage") or {}
    timings = r.get("timings") or {}
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tps": timings.get("prompt_per_second"),
        "decode_tps": timings.get("predicted_per_second"),
        "finish": r["choices"][0].get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30001")
    ap.add_argument("--model", default="")
    ap.add_argument("--log", default="", help="the lane's log; scanned for faults at the end")
    ap.add_argument("--ladder", default="16522:33022:33390",
                    help="the prompt sizes of R1:R2:R3 (default 16522:33022:33390, the Hy4 shape)")
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()

    base = a.base.rstrip("/")
    model = a.model
    if not model:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
            model = json.loads(r.read().decode())["data"][0]["id"]
    print(f"{base}  model {model}", flush=True)

    try:
        r1, r2, r3 = (int(x) for x in a.ladder.split(":"))
    except ValueError:
        sys.exit("--ladder must be R1:R2:R3, for example 16522:33022:33390")

    ok = True
    text = text_of(base, model, r1)
    parts = []
    for name, size in (("R1", r1), ("R2", r2), ("R3", r3)):
        want = size - count_tokens(base, model, text)
        if want > 0:
            text = text + text_of(base, model, want)
        res = ask(base, model, text, a.max_tokens)
        print(f"{name}  prompt {res['prompt_tokens']:>7} tokens  {res['wall_s']:>7.2f} s  "
              f"read {res['prompt_tps']}  decode {res['decode_tps']}  ({res['finish']})", flush=True)
        parts.append(res)
        if res["completion_tokens"] != a.max_tokens:
            ok = False
            print(f"  {name}: expected {a.max_tokens} completion tokens, got {res['completion_tokens']}")

    if a.log and os.path.exists(a.log):
        text_log = open(a.log, errors="replace").read()
        faults = [ln for ln in text_log.splitlines()
                  if re.search(r"illegal memory|err=|error:", ln, re.I)]
        print(f"log {a.log}: {len(faults)} fault-like lines")
        for ln in faults[-5:]:
            print("  " + ln.strip()[:160])
        if faults:
            ok = False
    else:
        print("log not scanned (no --log, or the file does not exist)")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
