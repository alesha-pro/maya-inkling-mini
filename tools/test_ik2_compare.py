"""tools/test_ik2_compare.py - the parity gate's arithmetic on synthetic dumps (CPU only, no model)."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ik2_compare as C  # noqa: E402

V = 64


def logsoftmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


class Pair:
    """a reference seams directory and an engine dump directory of T positions"""

    def __init__(self, root: pathlib.Path, T: int, noise: float = 0.0, flips=(), seed: int = 0):
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 3, (T, V))
        top = logits.argmax(-1)
        logits[np.arange(T), top] += 2.0                       # a clear margin everywhere
        ref = logsoftmax(logits)
        eng_logits = logits + rng.normal(0, noise, logits.shape) if noise else logits.copy()
        for t, near in flips:                                  # make the engine pick another token at t
            second = np.argsort(logits[t])[-2]
            if near:                                           # a near tie in the reference itself
                logits[t, second] = logits[t, top[t]] - 0.01
                ref[t] = logsoftmax(logits[t])
                eng_logits[t] = logits[t].copy()
            eng_logits[t, second] = eng_logits[t, top[t]] + (0.02 if near else 1.0)   # a real flip: the engine sure
        self.ref_dir, self.eng_dir = root / f"ref{seed}", root / f"eng{seed}"
        self.ref_dir.mkdir()
        self.eng_dir.mkdir()
        np.save(self.ref_dir / "logprobs.npy", ref)
        logsoftmax(eng_logits).astype(np.float32).tofile(self.eng_dir / "logprobs.f32")


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, *pairs, extra=()):
        args = [str(x) for p in pairs for x in (p.eng_dir, p.ref_dir)]
        return C.main(args + ["--gate", "--min-positions", "100", *extra])

    def test_identical_passes(self):
        self.assertEqual(self.run_gate(Pair(self.root, 200)), 0)

    def test_too_few_positions_fails(self):
        p = Pair(self.root, 50)
        self.assertEqual(C.main([str(p.eng_dir), str(p.ref_dir), "--gate"]), 1)   # default min 1000

    def test_near_tie_flip_tolerated_real_flip_fails(self):
        near = Pair(self.root, 200, flips=[(5, True)], seed=1)
        self.assertEqual(self.run_gate(near), 0)
        real = Pair(self.root, 200, flips=[(5, False)], seed=2)
        self.assertEqual(self.run_gate(real), 1)

    def test_engine_unsure_flip_is_near_tie(self):
        p = Pair(self.root, 200, seed=8)
        ref = np.load(p.ref_dir / "logprobs.npy")
        eng = np.fromfile(p.eng_dir / "logprobs.f32", np.float32).reshape(ref.shape).copy()
        t, top = 7, ref[7].argmax()
        eng[7, (top + 1) % V] = eng[7, top] + 0.01                 # the reference sure, the engine barely flipped
        eng[7] = logsoftmax(eng[7].astype(np.float64)).astype(np.float32)
        eng.tofile(p.eng_dir / "logprobs.f32")
        s = C.summarize([C.logprob_rows(p.eng_dir, C.load_ref(p.ref_dir))], 0.1)
        self.assertEqual((s["flips"], s["real_flips"]), (1, 0))

    def test_pairs_pool(self):
        a, b = Pair(self.root, 60, seed=3), Pair(self.root, 60, seed=4)
        self.assertEqual(self.run_gate(a, b), 0)                              # 120 >= 100 pooled

    def test_large_noise_fails_on_kl(self):
        self.assertEqual(self.run_gate(Pair(self.root, 200, noise=2.0, seed=5)), 1)

    def test_compact_reference_matches_full(self):
        p = Pair(self.root, 150, noise=0.05, seed=6)
        npz = self.root / "ref.npz"
        C.compact(p.ref_dir, npz, topk=V)                                     # top-K = vocab: exact
        full = C.summarize([C.logprob_rows(p.eng_dir, C.load_ref(p.ref_dir))], 0.1)
        comp = C.summarize([C.logprob_rows(p.eng_dir, C.load_ref(npz))], 0.1)
        self.assertEqual(full["top1"], comp["top1"])
        self.assertAlmostEqual(full["kl_mean"], comp["kl_mean"], places=5)
        self.assertEqual(C.main([str(p.eng_dir), str(npz), "--gate", "--min-positions", "100"]), 0)

    def test_shape_mismatch_is_input_error(self):
        p = Pair(self.root, 100, seed=7)
        np.zeros(10, np.float32).tofile(p.eng_dir / "logprobs.f32")
        self.assertEqual(C.main([str(p.eng_dir), str(p.ref_dir), "--gate"]), 2)

    def test_offset_raw_logits(self):
        # the decode/block modes: raw logits from position `first` on, compared with the reference's rows from there
        p = Pair(self.root, 300, seed=9)
        ref = np.load(p.ref_dir / "logprobs.npy")
        first = 120
        d = self.root / "dec"
        d.mkdir()
        (ref[first:] * 1.0 + 7.5).astype(np.float32).tofile(d / "logits.f32")   # unnormalized: shifted by a constant
        (d / "first_pos").write_text(f"{first}\n")
        s = C.summarize([C.logprob_rows(d, C.load_ref(p.ref_dir))], 0.1)
        self.assertEqual((s["positions"], s["flips"]), (300 - first, 0))
        self.assertLess(s["kl_mean"], 1e-6)
        npz = self.root / "ref9.npz"
        C.compact(p.ref_dir, npz, topk=V)
        self.assertEqual(C.main([str(d), str(npz), "--gate", "--min-positions", "100"]), 0)
        (d / "first_pos").write_text("100\n")                                  # rows that do not line up: bad input
        self.assertEqual(C.main([str(d), str(p.ref_dir), "--gate", "--min-positions", "100"]), 2)

    def test_wilson(self):
        lo, hi = C.wilson(45, 47)
        self.assertLess(lo, 0.87)
        self.assertGreater(hi, 0.98)


if __name__ == "__main__":
    unittest.main()
