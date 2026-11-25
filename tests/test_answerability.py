import unittest

from backend.rag.answerability import (
    CalibrationExample,
    CalibratedRetrievalGate,
)


class CalibratedRetrievalGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = CalibratedRetrievalGate.fit(
            [
                CalibrationExample(0.90, True),
                CalibrationExample(0.80, False),
                CalibrationExample(0.75, True),
                CalibrationExample(0.20, False),
            ],
            minimum_precision=1.0,
            calibration_id="test-calibration",
        )

    def test_selects_cutoff_from_labeled_examples(self):
        self.assertEqual(self.gate.threshold, 0.9)
        self.assertEqual(self.gate.empirical_precision, 1.0)

        supported = self.gate.assess(
            "question",
            [{"score": 0.95}],
        )
        unsupported = self.gate.assess(
            "question",
            [{"score": 0.80}],
        )

        self.assertTrue(supported.sufficient)
        self.assertEqual(supported.calibration_id, "test-calibration")
        self.assertFalse(unsupported.sufficient)
        self.assertEqual(unsupported.reason, "below_calibrated_cutoff")

    def test_missing_score_fails_closed(self):
        assessment = self.gate.assess("question", [{"text": "evidence"}])
        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.reason, "retrieval_score_missing")

    def test_fit_requires_both_labels(self):
        with self.assertRaisesRegex(ValueError, "unanswerable"):
            CalibratedRetrievalGate.fit(
                [CalibrationExample(0.9, True)]
            )

    def test_fit_requires_valid_precision_target(self):
        with self.assertRaisesRegex(ValueError, "minimum_precision"):
            CalibratedRetrievalGate.fit(
                [
                    CalibrationExample(0.9, True),
                    CalibrationExample(0.1, False),
                ],
                minimum_precision=0,
            )


if __name__ == "__main__":
    unittest.main()
