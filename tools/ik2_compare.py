"""tools/ik2_compare.py - maya-ik2's dump (--dump DIR: blk.L.out.f32, logprobs.f32) against ref/inkling2.py's seams
(--dump-dir: blk.L.out.npy, logprobs.npy), the same tokens through both - and, with --gate, a parity gate.

    python tools/ik2_compare.py ENGINE_DUMP_DIR REF [ENGINE_DUMP_DIR REF ...] [--gate] [--json OUT]
    python tools/ik2_compare.py --compact REF_SEAMS_DIR OUT.npz [--topk 256]

ENGINE_DUMP_DIR may instead hold logits.f32 (IK2_LOGITS_OUT's raw logits) and first_pos, the position its first row
is (the decode and block modes of bench/parity/run_gate.sh: the resident one-token path and the drafter's verify
windows, which --dump does not take): those rows are compared against the reference's from there.

REF is either ref/inkling2.py's seams directory or a compact reference (.npz, written by --compact): the reference's
top-K log-probabilities a position, small enough to keep in the repository, so the gate needs neither the float64
reference run nor its multi-GB dump. Several pairs (several texts) are pooled into one verdict.

Per layer (full reference directories only): the relative error of its output (|a - b| / |b| over [T][C]), the worst
token's cosine, attn_out's error and the experts picked. Over the log-probabilities:
  top-1     the share of positions where both pick the same token, with its Wilson 95% interval;
  flips     positions where they differ, split into near ties (the reference's top-2 margin, or the engine's between
            its top and the reference's, < --tie nats: one side unsure - a router near tie cascading through the layers
            does this, expected) and real ones (both sure, of different tokens: a bug until shown otherwise);
  KL        KL(ref || engine) a position: mean, p99 and max (with a compact reference: over the reference's top-K,
            whose mass is reported - the tail it leaves out adds at most its own mass x its log-ratio);
  dlogp     max |engine - ref| of the reference's top token's log-probability.

--gate checks them against the limits below (each flag overrides one) and exits 1 when any fails, 2 on bad input.
The defaults are a regression gate: the engine as measured on bench/parity's prompts when the gate was made, with
headroom (see LIMITS). Its flips today are router near ties cascading through the loop (an expert pick differs a few
visits before the flip), not a broken kernel; a kernel bug moves KL and top-1 well past these. Use >= 1000 positions:
with 47, top-1's 95% interval is about [0.86, 0.99]; --min-positions enforces it.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

LIMITS = {                       # the gate's defaults: bench/parity's first run (2026-11-25, cc41ab1, FP16 KV) with headroom
    "min_positions": 1000,
    "top1_min": 0.96,            # measured 0.978 (95% CI 0.970..0.983) over 2048 positions
    "real_flips_max": 15,        # measured 7, all after a router pick that already differed at that token
    "kl_mean_max": 6e-3,         # measured 3.1e-3
    "kl_p99_max": 6e-2,          # measured 3.0e-2 (3.4e-2 against the compact reference)
    "kl_max_max": 0.4,           # measured 0.18
    "dlogp_max": 1.0,            # measured 0.56
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """the Wilson score interval of k successes in n (95% by default)"""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---- the reference, full or compact
def load_ref(path: pathlib.Path) -> dict:
    """{'ids': [T][K] int64, 'lp': [T][K] float64} sorted descending, or {'full': [T][V]} for a seams directory"""
    if path.suffix == ".npz":
        z = np.load(path)
        return {"ids": z["ids"].astype(np.int64), "lp": z["lp"].astype(np.float64), "dir": None}
    return {"full": np.load(path / "logprobs.npy").astype(np.float64), "dir": path}


def compact(ref_dir: pathlib.Path, out: pathlib.Path, topk: int) -> None:
    r = np.load(ref_dir / "logprobs.npy").astype(np.float64)
    k = min(topk, r.shape[1])
    ids = np.argpartition(-r, k - 1, axis=-1)[:, :k]
    lp = np.take_along_axis(r, ids, -1)
    o = np.argsort(-lp, axis=-1)
    ids, lp = np.take_along_axis(ids, o, -1), np.take_along_axis(lp, o, -1)
    np.savez_compressed(out, ids=ids.astype(np.int32), lp=lp.astype(np.float32), n_vocab=np.int64(r.shape[1]))
    mass = np.exp(lp).sum(-1)
    print(f"{out}: {r.shape[0]} positions, top-{k}, reference mass covered min {mass.min():.6f} mean {mass.mean():.6f}")


# ---- one pair
def layers(E: pathlib.Path, R: pathlib.Path) -> list[dict]:
    out = []
    L = 0
    while (R / f"blk.{L}.out.npy").exists():
        r = np.load(R / f"blk.{L}.out.npy").astype(np.float64)
        f = E / f"blk.{L}.out.f32"
        if not f.exists():
            print(f"layer {L}: no engine dump")
            break
        e = np.fromfile(f, np.float32).astype(np.float64).reshape(r.shape)
        row = {"layer": L, "rel_err": float(np.linalg.norm(e - r) / np.linalg.norm(r)),
               "worst_cos": float(((e * r).sum(-1) / (np.linalg.norm(e, axis=-1) * np.linalg.norm(r, axis=-1))).min())}
        extra = ""
        fa, ra = E / f"blk.{L}.attn_out.f32", R / f"blk.{L}.attn_out.npy"
        if fa.exists() and ra.exists():
            rr = np.load(ra).astype(np.float64)
            ea = np.fromfile(fa, np.float32).astype(np.float64).reshape(rr.shape)
            row["attn_out_rel_err"] = float(np.linalg.norm(ea - rr) / np.linalg.norm(rr))
            extra += f", attn_out rel err {row['attn_out_rel_err']:.2e}"
        fp, rp = E / f"blk.{L}.picks.i32", R / f"blk.{L}.picks.npy"
        if fp.exists() and rp.exists():
            rpk = np.load(rp).astype(np.int64)
            epk = np.fromfile(fp, np.int32).astype(np.int64).reshape(rpk.shape)
            row["experts_same"] = float(np.mean([len(set(a) & set(b)) for a, b in zip(rpk, epk)]))
            row["tokens_differ"] = int(np.sum([set(a) != set(b) for a, b in zip(rpk, epk)]))
            extra += f", experts same {row['experts_same']:.2f}/{rpk.shape[1]} ({row['tokens_differ']} tokens differ)"
        print(f"layer {L:2d}: rel err {row['rel_err']:.2e}, worst token cos {row['worst_cos']:.6f}{extra}")
        out.append(row)
        L += 1
    return out


def engine_rows(E: pathlib.Path, V: int | None, T: int) -> tuple[np.ndarray, int]:
    """the engine's log-probabilities [n][V] and the position of its first row: logprobs.f32 (--dump), or logits.f32
    (IK2_LOGITS_OUT: raw logits, normalized here) with first_pos - the decode and block modes, whose rows start
    past a prefill"""
    first = int((E / "first_pos").read_text()) if (E / "first_pos").exists() else 0
    raw = not (E / "logprobs.f32").exists() and (E / "logits.f32").exists()
    f = E / ("logits.f32" if raw else "logprobs.f32")
    e = np.fromfile(f, np.float32)
    n = T - first if V is None else None
    if V is None:
        if n <= 0 or e.size % n:
            raise ValueError(f"{f}: {e.size} values, not {n} positions of one vocab")
        V = e.size // n
    if e.size % V:
        raise ValueError(f"{f}: {e.size} values, not a multiple of the vocab {V}")
    e = e.reshape(-1, V)
    if first + e.shape[0] != T:
        raise ValueError(f"{E}: {e.shape[0]} positions from {first}, against the reference's {T}")
    if raw:
        e = e.astype(np.float64)
        m = e.max(-1, keepdims=True)
        e = e - m - np.log(np.exp(e - m).sum(-1, keepdims=True))
    return e, first


def logprob_rows(E: pathlib.Path, ref: dict) -> dict:
    """per-position arrays for one pair: agree, margin, kl, dlogp, mass"""
    if "full" in ref:
        r = ref["full"]
        e, first = engine_rows(E, r.shape[1], r.shape[0])
        e, r = e.astype(np.float64), r[first:]
        top_r, top_e = r.argmax(-1), e.argmax(-1)
        s = np.sort(r, -1)
        margin = s[:, -1] - s[:, -2]
        pr = np.exp(r)
        kl = (pr * (r - e)).sum(-1)
        mass = np.ones(r.shape[0])
        d = np.abs(np.take_along_axis(e, top_r[:, None], -1) - np.take_along_axis(r, top_r[:, None], -1))[:, 0]
        emargin = (np.take_along_axis(e, top_e[:, None], -1) - np.take_along_axis(e, top_r[:, None], -1))[:, 0]
    else:
        e, first = engine_rows(E, None, ref["ids"].shape[0])
        ids, lpr = ref["ids"][first:], ref["lp"][first:]
        top_r, top_e = ids[:, 0], e.argmax(-1)
        margin = lpr[:, 0] - lpr[:, 1]
        le = np.take_along_axis(e, ids, -1).astype(np.float64)
        pr = np.exp(lpr)
        kl = (pr * (lpr - le)).sum(-1)
        mass = pr.sum(-1)
        d = np.abs(le[:, 0] - lpr[:, 0])
        emargin = e.max(-1).astype(np.float64) - le[:, 0]
    return {"agree": top_r == top_e, "margin": margin, "emargin": emargin, "kl": kl, "dlogp": d, "mass": mass}


def summarize(rows: list[dict], tie: float) -> dict:
    agree = np.concatenate([r["agree"] for r in rows])
    margin = np.concatenate([r["margin"] for r in rows])
    emargin = np.concatenate([r["emargin"] for r in rows])
    kl = np.concatenate([r["kl"] for r in rows])
    d = np.concatenate([r["dlogp"] for r in rows])
    mass = np.concatenate([r["mass"] for r in rows])
    n, k = int(agree.size), int(agree.sum())
    flips = ~agree
    real = flips & (margin >= tie) & (emargin >= tie)     # both sure, of different tokens
    lo, hi = wilson(k, n)
    return {
        "positions": n, "top1": k / n if n else 0.0, "top1_ci95": [lo, hi],
        "flips": int(flips.sum()), "near_tie_flips": int((flips & ~real).sum()), "real_flips": int(real.sum()),
        "kl_mean": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)), "kl_max": float(kl.max()),
        "dlogp_max": float(d.max()), "ref_mass_min": float(mass.min()),
    }


def gate(s: dict, lim: dict) -> list[str]:
    fails = []
    if s["positions"] < lim["min_positions"]:
        fails.append(f"positions {s['positions']} < {lim['min_positions']}")
    if s["top1"] < lim["top1_min"]:
        fails.append(f"top-1 {s['top1']:.4f} < {lim['top1_min']}")
    if s["real_flips"] > lim["real_flips_max"]:
        fails.append(f"real flips {s['real_flips']} > {lim['real_flips_max']}")
    for key in ("kl_mean", "kl_p99", "kl_max"):
        if s[key] > lim[key + "_max"]:
            fails.append(f"{key} {s[key]:.3e} > {lim[key + '_max']:.3e}")
    if s["dlogp_max"] > lim["dlogp_max"]:
        fails.append(f"max |dlogp| {s['dlogp_max']:.3e} > {lim['dlogp_max']:.3e}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pairs", nargs="*", help="ENGINE_DUMP_DIR REF, repeated")
    ap.add_argument("--compact", nargs=2, metavar=("REF_SEAMS_DIR", "OUT_NPZ"))
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--gate", action="store_true", help="exit 1 when a limit fails")
    ap.add_argument("--tie", type=float, default=0.1, help="a flip is a near tie below this reference margin (nats)")
    ap.add_argument("--json", help="write the summary (and the verdict) here")
    ap.add_argument("--width", type=int, default=4608, help=argparse.SUPPRESS)   # kept for old command lines
    for key, v in LIMITS.items():
        ap.add_argument("--" + key.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)

    if a.compact:
        compact(pathlib.Path(a.compact[0]), pathlib.Path(a.compact[1]), a.topk)
        return 0
    if not a.pairs or len(a.pairs) % 2:
        ap.error("give ENGINE_DUMP_DIR REF pairs")

    rows, per_layer = [], []
    try:
        for i in range(0, len(a.pairs), 2):
            E, R = pathlib.Path(a.pairs[i]), pathlib.Path(a.pairs[i + 1])
            ref = load_ref(R)
            if ref["dir"] is not None and not (E / "first_pos").exists():
                per_layer.append({"engine": str(E), "layers": layers(E, ref["dir"])})
            rows.append(logprob_rows(E, ref))
            one = summarize(rows[-1:], a.tie)
            print(f"{E}: {one['positions']} positions, top-1 {one['top1'] * 100:.1f}%, KL mean {one['kl_mean']:.3e}")
    except (OSError, ValueError, KeyError) as e:
        print(f"ik2_compare: {e}", file=sys.stderr)
        return 2

    s = summarize(rows, a.tie)
    lo, hi = s["top1_ci95"]
    print(f"logprobs: {s['positions']} positions, top-1 agree {s['top1'] * 100:.1f}% (95% CI {lo * 100:.1f}..{hi * 100:.1f}), "
          f"flips {s['flips']} ({s['near_tie_flips']} near ties < {a.tie} nats on a side, {s['real_flips']} real), "
          f"max |dlogp| of the reference's top {s['dlogp_max']:.3e}, "
          f"KL mean {s['kl_mean']:.3e} p99 {s['kl_p99']:.3e} max {s['kl_max']:.3e}"
          + (f", reference mass covered >= {s['ref_mass_min']:.4f}" if s["ref_mass_min"] < 1 else ""))
    lim = {k: getattr(a, k) for k in LIMITS}
    fails = gate(s, lim) if a.gate else []
    if a.gate:
        print("PARITY GATE: " + ("PASS" if not fails else "FAIL - " + "; ".join(fails)))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"summary": s, "limits": lim if a.gate else None,
                                                    "pass": (not fails) if a.gate else None, "fails": fails,
                                                    "layers": per_layer}, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
