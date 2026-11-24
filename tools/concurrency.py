#!/usr/bin/env python3
"""Concurrency ladder for a Basalt lane: C requests at once, per-request and aggregate decode.

Usage:
  concurrency.py BASE C [TOKENS MAX_TOKENS] [PROMPT]
With TOKENS the prompt is filler of about that many tokens; with PROMPT the text is sent as is
(use a prompt that asks for a long answer: short answers make the decode windows meaningless).
"""
import json
import sys
import threading
import time
import urllib.request

FILLER = ("The committee reviewed the quarterly figures and noted that several regional offices had "
          "exceeded their targets, while others reported delays caused by supply chain disruptions, "
          "staffing changes and unusually severe weather. ")


def get_json(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post_json(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def count(base, model, text):
    try:
        return post_json(base, "/v1/messages/count_tokens",
                         {"model": model, "messages": [{"role": "user", "content": text}]})["input_tokens"]
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def one(base, model, text, max_tokens, i, barrier):
    body = {"model": model,
            "messages": [{"role": "user", "content": f"run {i}-{time.time_ns()}\n" + text}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    barrier.wait()
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                piece = ch.get("text") or d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
    n = usage["completion_tokens"]
    return {"i": i, "prompt": usage["prompt_tokens"], "tokens": n,
            "ttft_s": round((first or t0) - t0, 2),
            "decode_s": round((last - first) if (last and first) else 0, 2),
            "decode_tps": round((n - 1) / (last - first), 1) if (last and first and last > first) else None}


def run(conc, base, model, text, max_tokens, label):
    print(f"{label}  model {model}  prompt {count(base, model, text)} tokens  max_tokens {max_tokens}")
    rows, lock = [], threading.Lock()
    barrier = threading.Barrier(conc + 1)

    def worker(i):
        r = one(base, model, text, max_tokens, i, barrier)
        with lock:
            rows.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    barrier.wait()
    t_start = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    rows.sort(key=lambda r: r["i"])
    total = sum(r["tokens"] for r in rows)
    dec_s = sum(r["decode_s"] for r in rows) or 1e-9
    for r in rows:
        print(f"req{r['i']}  prompt {r['prompt']:>6}  {r['tokens']:>4} tok  ttft {r['ttft_s']:>6.2f}s  "
              f"{r['decode_tps']} tok/s")
    print(f"c={conc}  wall {wall:.1f}s  total {total} tok  "
          f"aggregate(decode only) {total/dec_s:.1f} tok/s  "
          f"aggregate(wall incl. prefill) {total/wall:.1f} tok/s  "
          f"mean per-request {total/dec_s/conc:.1f} tok/s")


def main():
    base = sys.argv[1].rstrip("/")
    conc = int(sys.argv[2])
    model = get_json(base, "/v1/models")["data"][0]["id"]
    if len(sys.argv) > 5:
        run(conc, base, model, sys.argv[5], int(sys.argv[4]), "custom prompt")
        return
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    probe = FILLER * 200
    per_char = count(base, model, probe) / max(len(probe), 1)
    text = FILLER * max(1, int(tokens / max(per_char * len(FILLER), 1e-9)))
    while count(base, model, text) < tokens:
        text += FILLER
    run(conc, base, model, text, max_tokens, f"filler {tokens}")


if __name__ == "__main__":
    main()
