#!/usr/bin/env python3
"""GSM8K against a running Maya server, one recipe: the pinned test split, a fixed prompt, greedy.

  python3 tools/gsm8k_bench.py run --url http://127.0.0.1:18000 --out bench/gsm8k/results/X.json
  python3 tools/gsm8k_bench.py run --n 96 ...            # the first 96 questions only (a quick look)
  python3 tools/gsm8k_bench.py compare A.json B.json     # paired: exact McNemar on the questions both ran

The questions: openai/grade-school-math's test.jsonl at commit b0bb162 (1319 questions, MIT), downloaded once to
data/gsm8k/ and checked by its SHA-256.  Each question goes alone as a user turn through /v1/chat/completions with
thinking off, temperature 0 and max_tokens 1024; the answer is the number after the last "####", else the last number
in the reply.  A reply cut by max_tokens counts as wrong and is marked "length".  The output JSON keeps the server's
/metrics engine block, every reply and its verdict, so two runs compare question by question.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "gsm8k" / "test.jsonl"
COMMIT = "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"
SOURCE = f"https://raw.githubusercontent.com/openai/grade-school-math/{COMMIT}/grade_school_math/data/test.jsonl"
SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
N_ALL = 1319

PROMPT = ("{question}\n\nSolve the problem step by step. Then write the final answer on its own last line, "
          "as a plain number, in the form:\n#### <number>")

NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def load_questions(path: Path = DATA) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE}", file=sys.stderr)
        with urllib.request.urlopen(SOURCE, timeout=60) as r:
            raw = r.read()
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise SystemExit(f"{SOURCE}: SHA-256 is not the pinned one")
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit(f"{path}: SHA-256 is not the pinned {SHA256} - delete it to download it again")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == N_ALL, len(rows)
    return rows


def norm(s: str | None) -> str | None:
    """A number as text without $, commas or spaces, and without a trailing .0: "1,000.00" -> "1000"."""
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip().rstrip(".")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def gold_of(answer: str) -> str:
    g = norm(answer.split("####")[-1])
    assert g is not None, answer
    return g


def extract(reply: str) -> str | None:
    """The number after the last "####", else the last number in the reply."""
    marked = re.findall(r"####\s*(?:\*\*)?\s*(" + NUM + ")", reply)
    if marked:
        return norm(marked[-1])
    nums = re.findall(NUM, reply)
    return norm(nums[-1]) if nums else None


def ask(url: str, key: str, question: str, max_tokens: int, timeout: float) -> dict:
    body = {"model": "maya", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": PROMPT.format(question=question)}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    ch = out["choices"][0]
    return {"reply": ch["message"].get("content") or "", "finish": ch.get("finish_reason"),
            "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
            "seconds": round(time.time() - t0, 2)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: the discordant pairs b (A right, B wrong) and c (A wrong, B right) under Bin(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def server_info(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
            return json.loads(r.read()).get("engine") or {}
    except (OSError, ValueError):
        return {}


def cmd_run(a: argparse.Namespace) -> int:
    qs = load_questions()
    idx = list(range(a.offset, min(N_ALL, a.offset + a.n)))
    info = server_info(a.url)
    print(f"{len(idx)} questions, server: {json.dumps(info)[:200]}", flush=True)
    rows: dict[int, dict] = {}
    t0 = time.time()

    def one(i: int) -> dict:
        q = qs[i]
        try:
            r = ask(a.url, a.api_key, q["question"], a.max_tokens, a.timeout)
        except (OSError, urllib.error.HTTPError, ValueError, KeyError) as e:
            return {"i": i, "error": str(e), "correct": False}
        pred, gold = extract(r["reply"]), gold_of(q["answer"])
        ok = pred is not None and pred == gold and r["finish"] != "length"
        return {"i": i, "gold": gold, "pred": pred, "correct": ok, **r}

    def save(final: bool) -> dict:
        done = [rows[i] for i in sorted(rows)]
        k = sum(r["correct"] for r in done)
        lo, hi = wilson(k, len(done))
        summary = {"n": len(done), "correct": k, "accuracy": round(k / len(done), 4) if done else None,
                   "wilson95": [round(lo, 4), round(hi, 4)],
                   "length_cut": sum(r.get("finish") == "length" for r in done),
                   "no_answer": sum(r.get("pred") is None for r in done),
                   "errors": sum("error" in r for r in done),
                   "completion_tokens": sum(r.get("completion_tokens") or 0 for r in done),
                   "seconds": round(time.time() - t0, 1), "complete": final}
        doc = {"recipe": {"dataset": SOURCE, "sha256": SHA256, "prompt": PROMPT, "max_tokens": a.max_tokens,
                          "temperature": 0, "thinking": False, "offset": a.offset, "n": len(idx),
                          "concurrency": a.concurrency},
               "server": info, "label": a.label, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "summary": summary, "rows": done}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        return summary

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for n, r in enumerate(ex.map(one, idx), 1):
            rows[r["i"]] = r
            if not r["correct"]:
                why = r.get("error") or ("cut at max_tokens" if r.get("finish") == "length"
                                         else f"{r.get('pred')} against {r.get('gold')}")
                print(f"  #{r['i']}: wrong ({why})", flush=True)
            if n % 25 == 0:
                s = save(False)
                print(f"{n}/{len(idx)}: {s['correct']} right, {s['seconds']:.0f} s", flush=True)
    s = save(True)
    print(f"{s['correct']} of {s['n']} ({100 * s['accuracy']:.1f} %, Wilson 95% {100 * s['wilson95'][0]:.1f}-"
          f"{100 * s['wilson95'][1]:.1f}); cut at max_tokens {s['length_cut']}, no number {s['no_answer']}, "
          f"errors {s['errors']}; {s['completion_tokens']} tokens in {s['seconds']:.0f} s")
    return 0 if s["errors"] == 0 else 1


def compare(a_doc: dict, b_doc: dict) -> dict:
    ra = {r["i"]: r["correct"] for r in a_doc["rows"] if "error" not in r}
    rb = {r["i"]: r["correct"] for r in b_doc["rows"] if "error" not in r}
    both = sorted(set(ra) & set(rb))
    b = sum(ra[i] and not rb[i] for i in both)
    c = sum(rb[i] and not ra[i] for i in both)
    return {"paired": len(both), "a_correct": sum(ra[i] for i in both), "b_correct": sum(rb[i] for i in both),
            "a_only": b, "b_only": c, "p_mcnemar": mcnemar_exact(b, c)}


def cmd_compare(a: argparse.Namespace) -> int:
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (a.a, a.b)]
    for p, d in zip((a.a, a.b), docs):
        if d["recipe"]["sha256"] != SHA256 or d["recipe"]["prompt"] != PROMPT:
            print(f"warning: {p} was run with another recipe", file=sys.stderr)
    r = compare(*docs)
    print(f"{r['paired']} questions in both: A {r['a_correct']}, B {r['b_correct']}; "
          f"only A right {r['a_only']}, only B right {r['b_only']}; exact McNemar p = {r['p_mcnemar']:.3g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--api-key", default="")
    r.add_argument("--n", type=int, default=N_ALL, help=f"how many questions (default all {N_ALL})")
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--concurrency", type=int, default=1, help="requests in flight (the server's --parallel)")
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--label", default="")
    r.add_argument("--out")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
