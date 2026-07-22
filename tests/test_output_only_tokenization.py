import unittest

from train_medmcqa_and_synthetic_text_to_sql import tokenize_supervised_example


class FakeTokenizer:
    def __init__(self, encodings, eos_token_id=99, truncation_side="right"):
        self.encodings = {text: list(token_ids) for text, token_ids in encodings.items()}
        self.eos_token_id = eos_token_id
        self.truncation_side = truncation_side
        self.calls = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": list(self.encodings[text])}


class SupervisedTokenizationTest(unittest.TestCase):
    def test_masks_prompt_and_supervises_target_plus_eos(self):
        tokenizer = FakeTokenizer({"prompt": [10, 11, 12], " target": [20, 21]})

        tokenized = tokenize_supervised_example(
            {"text": "prompt", "output": " target"}, tokenizer, max_length=16
        )

        self.assertEqual(tokenized["input_ids"], [10, 11, 12, 20, 21, 99])
        self.assertEqual(tokenized["labels"], [-100, -100, -100, 20, 21, 99])
        self.assertEqual(tokenized["attention_mask"], [1] * 6)
        self.assertEqual(tokenizer.calls, [("prompt", False), (" target", False)])

    def test_does_not_duplicate_existing_target_eos(self):
        tokenizer = FakeTokenizer({"prompt": [10], "target": [20, 99]})

        tokenized = tokenize_supervised_example(
            {"text": "prompt", "output": "target"}, tokenizer, max_length=8
        )

        self.assertEqual(tokenized["input_ids"], [10, 20, 99])
        self.assertEqual(tokenized["labels"], [-100, 20, 99])

    def test_truncates_prompt_before_target_on_the_right(self):
        tokenizer = FakeTokenizer({"prompt": [10, 11, 12, 13], "target": [20, 21, 22]})

        tokenized = tokenize_supervised_example(
            {"text": "prompt", "output": "target"}, tokenizer, max_length=6
        )

        self.assertEqual(tokenized["input_ids"], [10, 11, 20, 21, 22, 99])
        self.assertEqual(tokenized["labels"], [-100, -100, 20, 21, 22, 99])

    def test_left_truncation_keeps_prompt_suffix_and_full_target(self):
        tokenizer = FakeTokenizer(
            {"prompt": [10, 11, 12, 13], "target": [20]}, truncation_side="left"
        )

        tokenized = tokenize_supervised_example(
            {"text": "prompt", "output": "target"}, tokenizer, max_length=4
        )

        self.assertEqual(tokenized["input_ids"], [12, 13, 20, 99])
        self.assertEqual(tokenized["labels"], [-100, -100, 20, 99])

    def test_target_overflow_drops_prompt_and_preserves_terminal_eos(self):
        tokenizer = FakeTokenizer({"prompt": [10, 11], "target": [20, 21, 22, 23, 24]})

        tokenized = tokenize_supervised_example(
            {"text": "prompt", "output": "target"}, tokenizer, max_length=4
        )

        self.assertEqual(tokenized["input_ids"], [20, 21, 22, 99])
        self.assertEqual(tokenized["labels"], [20, 21, 22, 99])
        self.assertEqual(tokenized["attention_mask"], [1, 1, 1, 1])

    def test_rejects_empty_target(self):
        tokenizer = FakeTokenizer({"prompt": [10]})

        with self.assertRaisesRegex(ValueError, "non-empty output target"):
            tokenize_supervised_example({"text": "prompt", "output": ""}, tokenizer, max_length=4)


if __name__ == "__main__":
    unittest.main()
