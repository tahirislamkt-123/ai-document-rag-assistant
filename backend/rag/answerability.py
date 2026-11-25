import hashlib
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CalibrationExample:
    score: float
    answerable: bool


@dataclass(frozen=True)
class AnswerabilityAssessment:
    sufficient: bool
    empirical_precision: Optional[float]
    calibration_id: str
    reason: Optional[str] = None


class CalibratedRetrievalGate:
    def __init__(
        self,
        threshold,
        empirical_precision,
        calibration_id,
        minimum_precision,
    ):
        self.threshold = threshold
        self.empirical_precision = empirical_precision
        self.calibration_id = calibration_id
        self.minimum_precision = minimum_precision

    @classmethod
    def fit(cls, examples, minimum_precision=1.0, calibration_id=None):
        examples = tuple(examples)
        if not 0 < minimum_precision <= 1:
            raise ValueError("minimum_precision must be in (0, 1]")
        if not examples:
            raise ValueError("At least one calibration example is required")
        if not any(item.answerable for item in examples):
            raise ValueError("Calibration requires answerable examples")
        if all(item.answerable for item in examples):
            raise ValueError("Calibration requires unanswerable examples")

        for item in examples:
            if not math.isfinite(item.score):
                raise ValueError("Calibration scores must be finite")

        positives = sum(item.answerable for item in examples)
        thresholds = sorted({item.score for item in examples}, reverse=True)
        candidates = []
        for threshold in thresholds:
            predicted = [item for item in examples if item.score >= threshold]
            true_positives = sum(item.answerable for item in predicted)
            precision = true_positives / len(predicted)
            recall = true_positives / positives
            if precision >= minimum_precision and true_positives:
                candidates.append((recall, precision, threshold))

        if candidates:
            _, empirical_precision, threshold = max(candidates)
        else:
            threshold = math.inf
            empirical_precision = None

        if calibration_id is None:
            fingerprint = "\n".join(
                "{:.12g}:{}".format(item.score, int(item.answerable))
                for item in sorted(
                    examples,
                    key=lambda item: (item.score, item.answerable),
                )
            )
            calibration_id = hashlib.sha256(
                fingerprint.encode("utf-8")
            ).hexdigest()[:12]

        return cls(
            threshold=threshold,
            empirical_precision=empirical_precision,
            calibration_id=calibration_id,
            minimum_precision=minimum_precision,
        )

    def assess(self, question, hits):
        del question
        if not hits:
            return AnswerabilityAssessment(
                sufficient=False,
                empirical_precision=self.empirical_precision,
                calibration_id=self.calibration_id,
                reason="no_retrieval_hits",
            )

        score = hits[0].get("score")
        if score is None:
            return AnswerabilityAssessment(
                sufficient=False,
                empirical_precision=self.empirical_precision,
                calibration_id=self.calibration_id,
                reason="retrieval_score_missing",
            )
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Retrieval score must be finite")

        sufficient = score >= self.threshold
        return AnswerabilityAssessment(
            sufficient=sufficient,
            empirical_precision=self.empirical_precision,
            calibration_id=self.calibration_id,
            reason=None if sufficient else "below_calibrated_cutoff",
        )
