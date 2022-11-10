import json

context_path = "./data/context.json"
train_path = "./data/train.json"
valid_path = "./data/valid.json"
test_path = "./data/test.json"

ts = ["train", "valid"]

context = json.load(open(context_path, 'r'))
train = json.load(open(train_path, 'r'))
valid = json.load(open(valid_path, 'r'))
test = json.load(open(test_path, 'r'))

# MC
mc_train = [{
    "id": j["id"],
    "sent1": j["question"],
    "sent2": j["question"],
    "label": j["paragraphs"].index(j["relevant"]),
    **dict([(f"ending{i}", context[int(j["paragraphs"][i])]) for i in range(4)]),    
} for j in train]

mc_valid = [{
    "id": j["id"],
    "sent1": j["question"],
    "sent2": j["question"],
    "label": j["paragraphs"].index(j["relevant"]),
    **dict([(f"ending{i}", context[int(j["paragraphs"][i])]) for i in range(4)]),    
} for j in valid]

for t in ts:
    with open(f"./data/mc_{t}.json", 'w') as f:
        json.dump(globals()[f"mc_{t}"], f)

# QA
qa_train = [{
    "id": j["id"],
    "question": j["question"],
    "context": context[int(j["relevant"])],
    "answers": {"text": [j["answer"]["text"]], "answer_start": [int(j["answer"]["start"])]}
} for j in train]

qa_valid = [{
    "id": j["id"],
    "question": j["question"],
    "context": context[int(j["relevant"])],
    "answers": {"text": [j["answer"]["text"]], "answer_start": [int(j["answer"]["start"])]}
} for j in valid]

for t in ts:
    with open(f"./data/qa_{t}.json", 'w') as f:
        json.dump(globals()[f"qa_{t}"], f)

mcqa_test2 = [{
    "id": j["id"],
    "question": j["question"],
    "sent1": j["question"],
    "sent2": j["question"],
    "paragraphs": j["paragraphs"],
    **dict([(f"ending{i}", context[int(j["paragraphs"][i])]) for i in range(4)]),    
} for j in test]

with open("./data/mcqa_test_sent12.json", 'w') as f:
    json.dump(globals()["mcqa_test2"], f)