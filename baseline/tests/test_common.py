import unittest

import numpy as np

from baseline.src.common import upper_tail_evidence


class UpperTailEvidenceTest(unittest.TestCase):
    def test_is_monotone_and_finite(self):
        train = np.asarray([1.0, 2.0, 3.0, 4.0])
        query = np.asarray([-1.0, 1.0, 2.5, 4.0, 10.0])
        value = upper_tail_evidence(train, query)
        self.assertTrue(np.isfinite(value).all())
        self.assertTrue(np.all(np.diff(value) >= 0))


    def test_uses_finite_sample_smoothing(self):
        train = np.asarray([1.0, 2.0, 3.0])
        value = upper_tail_evidence(train, np.asarray([100.0]))
        self.assertTrue(np.allclose(value, np.log(4.0)))


if __name__ == "__main__":
    unittest.main()
