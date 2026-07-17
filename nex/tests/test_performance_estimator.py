import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import performance_estimator as pe


class PerformanceEstimatorTests(unittest.TestCase):
    def test_exact_nex_int4_expert_geometry(self):
        self.assertEqual(pe.EXPERT_PARAMS, 12_582_912)
        self.assertEqual(pe.EXPERT_BYTES_INT4, 6_316_032)
        self.assertEqual(pe.EXPERT_CALLS_PER_TOKEN, 600)
        self.assertAlmostEqual(pe.COLD_EXPERT_GB_PER_TOKEN, 3.7896192, places=7)
        self.assertAlmostEqual(pe.EXPERT_STORE_GB, 194.02850304, places=7)

    def test_disk_ceiling(self):
        self.assertAlmostEqual(
            pe.disk_ceiling_tps(1.0, 0.0),
            1.0 / pe.COLD_EXPERT_GB_PER_TOKEN,
            places=9,
        )
        self.assertGreater(pe.disk_ceiling_tps(5.0, 0.75), 5.0)
        self.assertTrue(math.isinf(pe.disk_ceiling_tps(5.0, 1.0)))

    def test_anchor_scaling_is_bounded(self):
        estimates = {item.name: pe.scale_anchor(item) for item in pe.ANCHORS}
        minimum = estimates["minimum streaming box"]
        self.assertGreaterEqual(minimum.low_tps, 0.13)
        self.assertLessEqual(minimum.high_tps, 0.31)
        full = estimates["six RTX 5090 full-resident"]
        self.assertGreater(full.low_tps, 14.0)
        self.assertLess(full.high_tps, 17.0)

    def test_report_distinguishes_estimates_from_measurements(self):
        report = pe.report()
        self.assertIn("calibrated_anchors", report)
        self.assertIn("projected_tiers", report)
        self.assertIn("A real benchmark result overrides every estimate.", report["rules"])


if __name__ == "__main__":
    unittest.main()
