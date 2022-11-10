import json
import argparse

parser = argparse.ArgumentParser(description="Infer")
parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="",
    )
args = parser.parse_args()

context_path = args.context
test_path = args.test

context = json.load(open(context_path, 'r'))
test = json.load(open(test_path, 'r'))

mcqa_test2 = [{
    "id": j["id"],
    "question": j["question"],
    "sent1": j["question"],
    "sent2": j["question"],
    "paragraphs": j["paragraphs"],
    **dict([(f"ending{i}", context[int(j["paragraphs"][i])]) for i in range(4)]),    
} for j in test]

with open("./my_cache/mcqa_test_sent12.json", 'w') as f:
    json.dump(globals()["mcqa_test2"], f)