import unittest

from evaluation.evaluate_retrieval import aggregate_scores, score_ranked_ids


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_reward_relevant_document_at_rank_two(self):
        score = score_ranked_ids(
            ["irrelevant", "target", "other"],
            ["target"],
            k=3,
        )

        self.assertEqual(score["hit_at_1"], 0)
        self.assertEqual(score["hit_at_k"], 1)
        self.assertEqual(score["reciprocal_rank_at_k"], 0.5)
        self.assertEqual(score["first_relevant_rank"], 2)

    def test_relevant_document_outside_k_does_not_count(self):
        score = score_ranked_ids(
            ["one", "two", "three", "target"],
            ["target"],
            k=3,
        )

        self.assertEqual(score["hit_at_1"], 0)
        self.assertEqual(score["hit_at_k"], 0)
        self.assertEqual(score["reciprocal_rank_at_k"], 0)
        self.assertIsNone(score["first_relevant_rank"])

    def test_aggregates_query_level_scores(self):
        scores = [
            score_ranked_ids(["target"], ["target"]),
            score_ranked_ids(["other", "target"], ["target"]),
        ]

        self.assertEqual(
            aggregate_scores(scores),
            {
                "query_count": 2,
                "hit_at_1": 0.5,
                "hit_at_3": 1.0,
                "mrr_at_3": 0.75,
            },
        )

    def test_invalid_metric_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            score_ranked_ids([], ["target"], k=0)
        with self.assertRaises(ValueError):
            score_ranked_ids([], [])
        with self.assertRaises(ValueError):
            aggregate_scores([])
        with self.assertRaises(ValueError):
            aggregate_scores(
                [
                    score_ranked_ids(["target"], ["target"], k=1),
                    score_ranked_ids(["target"], ["target"], k=3),
                ]
            )


if __name__ == "__main__":
    unittest.main()
