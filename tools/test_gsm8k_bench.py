"""tools/gsm8k_bench.py: the answer's extraction, the statistics and a run against a fake server."""
import argparse
import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import gsm8k_bench as g


class Extract(unittest.TestCase):
    def test_marked(self):
        self.assertEqual(g.extract("16 - 3 - 4 = 9\n9 * 2 = 18\n#### 18"), "18")
        self.assertEqual(g.extract("#### $1,250.00"), "1250")
        self.assertEqual(g.extract("#### **42**"), "42")
        self.assertEqual(g.extract("#### -7"), "-7")
        self.assertEqual(g.extract("#### 0.50"), "0.5")
        self.assertEqual(g.extract("first #### 3 then\n#### 5"), "5")   # (the last mark)

    def test_unmarked_falls_back_to_last_number(self):
        self.assertEqual(g.extract("She makes 9 * 2 = $18 a day."), "18")
        self.assertEqual(g.extract("The answer is 1,000."), "1000")
        self.assertIsNone(g.extract("I cannot tell."))

    def test_gold(self):
        self.assertEqual(g.gold_of("Janet sells 9 eggs.\n#### 18"), "18")
        self.assertEqual(g.gold_of("x\n#### 1,000"), "1000")

    def test_norm(self):
        self.assertEqual(g.norm("18.0"), "18")
        self.assertEqual(g.norm("-0"), "0")
        self.assertIsNone(g.norm("abc"))


class Stats(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(g.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(g.mcnemar_exact(5, 0), 2 / 32)
        self.assertAlmostEqual(g.mcnemar_exact(0, 5), 2 / 32)
        # b=10, c=2: 2 * P(X <= 2), X ~ Bin(12, 1/2) = 2 * 79 / 4096
        self.assertAlmostEqual(g.mcnemar_exact(10, 2), 2 * 79 / 4096)
        self.assertEqual(g.mcnemar_exact(4, 4), 1.0)

    def test_wilson(self):
        lo, hi = g.wilson(89, 96)
        self.assertTrue(0.85 < lo < 0.86 and 0.96 < hi < 0.97, (lo, hi))

    def test_compare_pairs_by_question(self):
        a = {"rows": [{"i": 0, "correct": True}, {"i": 1, "correct": True}, {"i": 2, "correct": False},
                      {"i": 3, "correct": True}]}
        b = {"rows": [{"i": 0, "correct": True}, {"i": 1, "correct": False}, {"i": 2, "correct": True},
                      {"i": 9, "correct": True}, {"i": 3, "error": "x", "correct": False}]}
        r = g.compare(a, b)
        self.assertEqual((r["paired"], r["a_only"], r["b_only"]), (3, 1, 1))


class Dataset(unittest.TestCase):
    def test_rejects_another_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text('{"question": "q", "answer": "#### 1"}\n')
            with self.assertRaises(SystemExit):
                g.load_questions(p)


class FakeServer(http.server.BaseHTTPRequestHandler):
    replies = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.seen.append(body)
        q = body["messages"][0]["content"].split("\n")[0]
        text, finish = FakeServer.replies[q]
        out = json.dumps({"choices": [{"message": {"content": text}, "finish_reason": finish}],
                          "usage": {"completion_tokens": 7}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps({"engine": {"engine": "fake", "kv": "fp8"}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Run(unittest.TestCase):
    def test_run_against_a_fake_server(self):
        qs = [{"question": "Q0", "answer": "#### 18"}, {"question": "Q1", "answer": "#### 5"},
              {"question": "Q2", "answer": "#### 3"}, {"question": "Q3", "answer": "#### 4"}]
        FakeServer.replies = {"Q0": ("so\n#### 18", "stop"), "Q1": ("#### 6", "stop"),
                              "Q2": ("3 + ... #### 3", "length"), "Q3": ("it is 4", "stop")}
        FakeServer.seen = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch.object(g, "load_questions", return_value=qs), \
                    mock.patch.object(g, "N_ALL", 4):
                out = Path(d) / "r.json"
                a = argparse.Namespace(url=f"http://127.0.0.1:{srv.server_address[1]}", api_key="", n=4, offset=0,
                                       max_tokens=1024, concurrency=2, timeout=30, label="t", out=str(out))
                self.assertEqual(g.cmd_run(a), 0)
                doc = json.loads(out.read_text())
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual([r["correct"] for r in doc["rows"]], [True, False, False, True])
        self.assertEqual(doc["summary"]["correct"], 2)
        self.assertEqual(doc["summary"]["length_cut"], 1)   # (a cut reply is wrong even with the right number)
        self.assertEqual(doc["server"], {"engine": "fake", "kv": "fp8"})
        body = FakeServer.seen[0]
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
