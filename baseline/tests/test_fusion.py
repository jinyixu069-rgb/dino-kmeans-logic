import unittest

import numpy as np

from baseline.src.common import SPLITS, upper_tail_evidence
from baseline.src.fuse import score_without_labels


class FusionTest(unittest.TestCase):
    def setUp(self):
        self.names = {
            "train_good": np.asarray(["000", "001", "002"]),
            "test_good": np.asarray(["010"]),
            "test_logical": np.asarray(["020"]),
            "test_structural": np.asarray(["030"]),
        }
        self.b = {}
        for split in SPLITS:
            self.b[f"{split}_names"] = self.names[split]
        self.b.update({
            "train_good_B0": np.asarray([1.0, 2.0, 3.0]),
            "train_good_composition": np.asarray([3.0, 1.0, 2.0]),
            "test_good_B0": np.asarray([1.5]),
            "test_logical_B0": np.asarray([4.0]),
            "test_structural_B0": np.asarray([2.5]),
            "test_good_composition": np.asarray([1.5]),
            "test_logical_composition": np.asarray([2.5]),
            "test_structural_composition": np.asarray([4.0]),
        })
        self.s0 = {
            "train_good": np.asarray([0.1, 0.2, 0.3]),
            "test_good": np.asarray([0.15]),
            "test_logical": np.asarray([0.35]),
            "test_structural": np.asarray([0.25]),
        }
        self.s1 = {
            "train_good": np.asarray([0.3, 0.1, 0.2]),
            "test_good": np.asarray([0.15]),
            "test_logical": np.asarray([0.25]),
            "test_structural": np.asarray([0.35]),
        }

    def test_exact_frozen_formula(self):
        _, value = score_without_labels(
            self.b, self.names, self.s0, self.names, self.s1
        )
        b0_train = upper_tail_evidence(self.b["train_good_B0"], self.b["train_good_B0"])
        pc_train = upper_tail_evidence(
            self.b["train_good_composition"], self.b["train_good_composition"]
        )
        expected_b = upper_tail_evidence(
            np.maximum(b0_train, pc_train),
            np.maximum(value["e_B0"], value["e_composition"]),
        )
        expected = expected_b + value["e_S0"] + value["e_S1"]
        self.assertTrue(np.allclose(value["final"], expected))

    def test_rejects_misaligned_inputs(self):
        bad_names = dict(self.names)
        bad_names["test_good"] = np.asarray(["wrong"])
        with self.assertRaisesRegex(ValueError, "filename mismatch"):
            score_without_labels(self.b, bad_names, self.s0, self.names, self.s1)


if __name__ == "__main__":
    unittest.main()
