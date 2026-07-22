import unittest

from generation_test_utils import format_text as format_task_text
from loraplusmseq.data import format_text as format_public_text


PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class NaturalReasoningFormattingCompatibilityTest(unittest.TestCase):
    def test_prompt_only_with_input_matches_head_alpaca_format(self):
        example = {
            "instruction": "Solve the problem carefully.",
            "input": "What is 2 + 2?",
            "output": "4",
        }

        formatted = format_public_text(dict(example), "natural_reasoning_20k", prompt_only=True)

        self.assertEqual(formatted["text"], PROMPT_INPUT.format_map(example))
        self.assertEqual(formatted["data_name"], "natural_reasoning_20k")

    def test_prompt_and_output_without_input_matches_head_alpaca_format(self):
        example = {
            "instruction": "Explain the result step by step.",
            "input": "",
            "output": "First reason, then answer.",
        }

        formatted = format_public_text(dict(example), "natural_reasoning_20k", prompt_only=False)

        self.assertEqual(formatted["text"], PROMPT_NO_INPUT.format_map(example) + example["output"])

    def test_public_formatter_ignores_unrelated_prompt_field(self):
        example = {
            "instruction": "Use the baseline instruction.",
            "input": "baseline input",
            "output": "baseline output",
            "prompt": "THIS MUST NOT OVERRIDE THE BASELINE FORMAT",
        }

        prompt_only = format_public_text(dict(example), "natural_reasoning_20k", prompt_only=True)
        with_output = format_public_text(dict(example), "natural_reasoning_20k", prompt_only=False)
        expected_prompt = PROMPT_INPUT.format_map(example)

        self.assertEqual(prompt_only["text"], expected_prompt)
        self.assertEqual(with_output["text"], expected_prompt + example["output"])


class TaskSpecificFormattingTest(unittest.TestCase):
    def test_task_formatter_uses_explicit_preformatted_prompt(self):
        example = {
            "instruction": "Compatibility-only instruction.",
            "input": "compatibility-only input",
            "output": " target",
            "prompt": "Exact task prompt:\n",
        }

        prompt_only = format_task_text(dict(example), "medmcqa", prompt_only=True)
        with_output = format_task_text(dict(example), "medmcqa", prompt_only=False)

        self.assertEqual(prompt_only["text"], example["prompt"])
        self.assertEqual(with_output["text"], example["prompt"] + example["output"])

    def test_task_formatter_falls_back_when_explicit_prompt_is_empty(self):
        example = {
            "instruction": "Fallback instruction.",
            "input": "",
            "output": "fallback output",
            "prompt": "",
        }

        formatted = format_task_text(dict(example), "medmcqa", prompt_only=True)

        self.assertEqual(formatted["text"], PROMPT_NO_INPUT.format_map(example))


if __name__ == "__main__":
    unittest.main()
