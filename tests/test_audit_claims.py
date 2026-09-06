"""Exercise the real code paths in tools/audit_claims.py."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import audit_claims as ac  # noqa: E402


class TestBounds(unittest.TestCase):
    def test_absurd_db_is_impossible(self):
        v = ac.audit_number("result.psnr_db", 340.0)
        self.assertIsNotNone(v)
        self.assertEqual(v[0], ac.Verdict.IMPOSSIBLE)

    def test_db_beyond_float64_range_is_suspect(self):
        v = ac.audit_number("metrics.snr_db", 140.0)
        self.assertIsNotNone(v)
        self.assertEqual(v[0], ac.Verdict.SUSPECT)

    def test_plausible_db_passes(self):
        self.assertIsNone(ac.audit_number("metrics.snr_db", 62.0))

    def test_million_to_one_lossless_is_impossible(self):
        v = ac.audit_number("codec.compression_ratio", 1_200_000.0)
        self.assertIsNotNone(v)
        self.assertEqual(v[0], ac.Verdict.IMPOSSIBLE)

    def test_modest_ratio_passes(self):
        self.assertIsNone(ac.audit_number("codec.compression_ratio", 4.2))

    def test_fractional_losslessness_is_suspect(self):
        v = ac.audit_number("roundtrip.lossless", 0.97)
        self.assertIsNotNone(v)
        self.assertEqual(v[0], ac.Verdict.SUSPECT)


class TestBaselines(unittest.TestCase):
    def test_entropy_of_uniform_bytes_is_eight(self):
        vals = [float(i % 256) for i in range(1024)]
        self.assertAlmostEqual(ac.shannon_bits_per_sample(vals), 8.0, places=6)

    def test_constant_data_has_zero_entropy_and_no_ceiling(self):
        self.assertEqual(ac.shannon_bits_per_sample([3.0] * 64), 0.0)
        self.assertIsNone(ac.entropy_floor_ratio([3.0] * 64, 8))

    def test_lzma_beats_entropy_floor_on_structured_data(self):
        # Highly structured: memoryless entropy says 8 bits/sample, but a
        # context coder must do far better. This is exactly the gap a claim
        # like "we beat the entropy bound" has to be judged against.
        vals = [float(i % 256) for i in range(4096)]
        mem = ac.entropy_floor_ratio(vals, 8)
        real = ac.lzma_floor_ratio(vals)
        self.assertAlmostEqual(mem, 8.0, places=4)
        self.assertGreater(real, mem, "lzma should beat the memoryless ceiling here")

    def test_random_data_compresses_near_one_to_one(self):
        # Uniform doubles carry ~52 bits of mantissa entropy each, so no
        # lossless coder should get much traction. (Integers cast to float do
        # NOT satisfy this -- their low mantissa bytes are all zero and lzma
        # happily gets ~1.8x. The auditor has to be fed real doubles.)
        import random

        rnd = random.Random(7)
        vals = [rnd.uniform(-1e6, 1e6) for _ in range(2048)]
        real = ac.lzma_floor_ratio(vals)
        self.assertIsNotNone(real)
        self.assertLess(real, 1.1)

    def test_integers_cast_to_float_stay_compressible(self):
        # The trap above, pinned so the baseline cannot silently lie.
        import random

        rnd = random.Random(7)
        vals = [float(rnd.getrandbits(64) % 10**9) for _ in range(2048)]
        self.assertGreater(ac.lzma_floor_ratio(vals), 1.5)


class TestDrivers(unittest.TestCase):
    def test_audit_json_flags_and_collects_baselines(self):
        payload = {
            "exp1": {"psnr_db": 340.0, "compression_ratio": 1.7, "lossless": 1.0},
            "exp5": {"dtype_whitening_floor_bps": 0.003, "snr_db": 141.0},
            "signal": [float(i % 256) for i in range(512)],
        }
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "exp.json")
            with open(p, "w") as fh:
                json.dump(payload, fh)
            rep = ac.audit_json(p)

        verdicts = {f["path"]: f["verdict"] for f in rep["flags"]}
        self.assertEqual(verdicts["exp1.psnr_db"], ac.Verdict.IMPOSSIBLE)
        self.assertEqual(verdicts["exp5.snr_db"], ac.Verdict.SUSPECT)
        self.assertEqual(len(rep["array_baselines"]), 1)
        self.assertEqual(rep["array_baselines"][0]["n"], 512)
        self.assertGreater(rep["array_baselines"][0]["lzma9_ratio"], 8.0)

    def test_missing_path_returns_exit_code_two(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ac.main(["/nonexistent/dir", "--json"])
        self.assertEqual(rc, 2)

    def test_impossible_claim_sets_exit_code_one(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as fh:
                json.dump({"psnr_db": 900.0}, fh)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = ac.main([p, "--json"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
