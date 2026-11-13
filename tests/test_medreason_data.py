import unittest

from loraplusmseq.medreason_data import format_medreason, tokenize_medreason


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]


class MedReasonDataTest(unittest.TestCase):
    def test_plain_format_and_response_mask(self):
        example = format_medreason({
            "question": "Question?", "reasoning": "Reason", "answer": "Answer",
            "options": "DO NOT APPEND",
        })
        self.assertEqual(example["prompt"], "Question?\n\n")
        self.assertEqual(example["response"], "## Thinking\n\nReason\n\n## Final Answer\n\nAnswer")
        item = tokenize_medreason(example, CharacterTokenizer(), 1000)
        n = len(example["prompt"])
        self.assertEqual(item["labels"][:n], [-100] * n)
        self.assertEqual(item["labels"][n:], item["input_ids"][n:])
        self.assertEqual(item["input_ids"][-1], 1)

    def test_left_truncation_preserves_final_answer_and_eos(self):
        example = {"prompt": "long question", "response": "reasoning then answer"}
        full = tokenize_medreason(example, CharacterTokenizer(), 1000)
        short = tokenize_medreason(example, CharacterTokenizer(), 8)
        self.assertEqual(short["input_ids"], full["input_ids"][-8:])
        self.assertEqual(short["labels"], full["labels"][-8:])
        self.assertEqual(short["attention_mask"], [1] * 8)

    def test_invalid_fields_rejected(self):
        with self.assertRaises(ValueError):
            format_medreason({"question": "q", "reasoning": "", "answer": "a"})


if __name__ == "__main__":
    unittest.main()
