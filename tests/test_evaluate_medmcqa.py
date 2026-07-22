import unittest

from evaluate_medmcqa import (
    evaluate_medmcqa,
    evaluate_medmcqa_metrics,
    extract_letter,
)


class MedMCQATest(unittest.TestCase):
    def test_extracts_explicit_and_anchored_options(self):
        cases = {
            "A": "A",
            "A\nThe reason follows.": "A",
            "(b)": "B",
            "C. Explanation": "C",
            "Final answer: D": "D",
            "Final answer: (C)": "C",
            "The correct answer is (C).": "C",
            "Answer is B because it is indicated.": "B",
            "Option: (D)": "D",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(extract_letter(response), expected)

    def test_last_explicit_answer_wins(self):
        response = "Answer: A was my first thought. After checking, final answer: (C)."
        self.assertEqual(extract_letter(response), "C")

    def test_does_not_extract_arbitrary_letters_from_prose(self):
        for response in (
            "A patient presents with fever.",
            "Vitamin A deficiency is unlikely.",
            "Because the finding is nonspecific.",
            "Cardiac output is normal.",
            "ABCD",
        ):
            with self.subTest(response=response):
                self.assertEqual(extract_letter(response), "")

    def test_metrics_report_invalid_predictions(self):
        rows = [
            {"response": "A\nreason", "answer": "A"},
            {"response": "Final answer: B", "cop": 1},
            {"response": "C.", "answer_index": "3"},
            {"response": "No option was provided", "answer": "D"},
        ]
        metrics, checked = evaluate_medmcqa_metrics(rows)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["valid_prediction_rate"], 0.75)
        self.assertEqual(metrics["invalid_count"], 1)
        self.assertFalse(checked[2]["flag"])
        self.assertFalse(checked[3]["prediction_valid"])

        score, compatibility_rows = evaluate_medmcqa(rows)
        self.assertEqual(score, metrics["accuracy"])
        self.assertEqual(compatibility_rows, checked)


if __name__ == "__main__":
    unittest.main()
