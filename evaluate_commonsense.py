"""Evaluate commonsense benchmark predictions."""

import copy
import json
import os
import re

import fire


def extract_answer(dataset, sentence: str) -> str:
    sentence_ = sentence.strip()
    if dataset == "boolq":
        pred_answers = re.findall(r"true|false", sentence_)
    elif dataset == "piqa":
        pred_answers = re.findall(r"solution1|solution2", sentence_)
    elif dataset in ["social_i_qa", "arc-challenge", "arc-easy", "openbookqa"]:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
    elif dataset == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
    elif dataset == "winogrande":
        pred_answers = re.findall(r"option1|option2", sentence_)
    else:
        pred_answers = []
    return pred_answers[0] if pred_answers else ""


def commonsense_acc(predict_file):
    test_dataset_l = [
        "boolq",
        "piqa",
        "social_i_qa",
        "hellaswag",
        "winogrande",
        "arc-challenge",
        "arc-easy",
        "openbookqa",
    ]
    result = {}
    for dataset in test_dataset_l:
        save_path = predict_file.replace("boolq", dataset)
        with open(save_path, "r", encoding="utf-8") as f:
            data_l = [json.loads(one) for one in f.readlines()]
        total = len(data_l)
        correct = 0
        check_path = os.path.join(os.path.dirname(save_path), f"{dataset}_predict_checkanswer.jsonl")
        if os.path.exists(check_path):
            os.remove(check_path)
        for data in data_l:
            label = data.get("answer")
            predict = extract_answer(dataset, data.get("response", ""))
            flag = label == predict
            if flag:
                correct += 1

            new_data = copy.deepcopy(data)
            new_data["pred"] = predict
            new_data["flag"] = flag
            with open(check_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
        result[dataset] = correct / total
        print(f"{dataset}: accuracy {correct} {correct / total}")

    acc_l = result.values()
    result["average"] = sum(acc_l) / len(result)
    return result


def main(predict_file: str):
    print(predict_file)
    result = commonsense_acc(predict_file)
    print(f"acc:{result}")
    result["predict_file"] = predict_file
    directory = os.path.dirname(os.path.dirname(predict_file))
    with open(os.path.join(directory, "acc_score.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)

