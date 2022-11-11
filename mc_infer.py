from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    BertTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from accelerate import Accelerator
import torch
import argparse
import json
from tqdm import tqdm
from itertools import chain
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="Infer")
parser.add_argument(
        "--mc_token_json",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="",
    )
parser.add_argument(
        "--mc_model",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--mc_config",
        type=str,
        default=None,
        help="",
    )

parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="",
    )

args = parser.parse_args()

def mc_preprocess_function(examples):
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    #label_column_name = "label" if "label" in column_names else "labels"
    if type(examples[context_name]) == str:
        first_sentences = [[examples[context_name]]*4]
    else:
        first_sentences = [[context] * 4 for context in examples[context_name]]
    if type(examples[context_name]) == str:
        question_headers = [examples[question_header_name]]
    else:
        question_headers = examples[question_header_name]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]
    #labels = examples[label_column_name]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = mc_tokenizer(
        first_sentences,
        second_sentences,
        max_length=512,
        padding="max_length",
        truncation=True,
        #return_tensors="pt",
    )
    # Un-flatten
    #print(tokenized_examples.keys())
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    #print(tokenized_inputs)
    return tokenized_inputs

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=2)

    mc_tokenizer = AutoTokenizer.from_pretrained(args.mc_token_json, use_fast=True)

    mc_config = AutoConfig.from_pretrained(args.mc_config)
    mc_model = AutoModelForMultipleChoice.from_config(mc_config)
    mc_model.load_state_dict(torch.load(args.mc_model))
    mc_model = accelerator.prepare(mc_model)

    test = json.load(open(args.test_file, 'r'))

    data_collator = default_data_collator
    d = Dataset.from_list(test)
    with accelerator.main_process_first():
        d = d.map(
                mc_preprocess_function, batched=True, remove_columns=d.column_names
            )
    eval_dataloader = DataLoader(d, collate_fn=data_collator, batch_size=args.batch)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    #print(d)

    mc_model.eval()
    output_labels = False
    if output_labels:
        out = open("my_sample_mc.csv", 'w')
        out.write("id,label\n")
    all_pred = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = mc_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions = accelerator.gather_for_metrics(predictions).cpu().numpy().tolist()
        all_pred+=predictions
    print("Done pred")
    #print(all_pred[:200])
    all_pred = accelerator.prepare(all_pred)

    print("lens", len(all_pred), len(test))
    for i, p in enumerate(all_pred):
        d = test[i]
        try:
            choice = p
        except:
            print(i)
            break
        if output_labels:
            out.write(f"{d['id']},{int(choice)}\n")
            continue
        d["context"] = d[f"ending{int(choice)}"]
        [d.pop(f) for f in ["sent1", "sent2", "ending0", "ending1", "ending2", "ending3"]]
    
    if not output_labels:
        json.dump(test, open(args.output, 'w'))
    print("finish")

