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
    EvalPrediction,
    DataCollatorWithPadding,
)
import torch
import argparse
import json
from tqdm import tqdm
from itertools import chain
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from utils_qa import postprocess_qa_predictions
from accelerate import Accelerator
import numpy as np


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
        "--qa_token_json",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--mc_model",
        type=str,
        default=None,
        help="",
    )
parser.add_argument(
        "--qa_model",
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
        "--qa_config",
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
parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="",
    )

args = parser.parse_args()


def qa_prepare_train_features(examples):
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    doc_stride = 128
    pad_on_right = qa_tokenizer.padding_side == "right"
    max_seq_length = min(512, qa_tokenizer.model_max_length)
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    if type(examples[question_column_name]) == str:
        examples[question_column_name] = examples[question_column_name].lstrip()
    else:
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    #print(examples[question_column_name if pad_on_right else context_column_name],
    #    examples[context_column_name if pad_on_right else question_column_name])
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = qa_tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    #print(tokenized_examples)
    return tokenized_examples.to("cuda")

def qa_prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    question_column_name = "question"
    context_column_name = "context"
    doc_stride = 128
    pad_on_right = qa_tokenizer.padding_side == "right"
    max_seq_length = min(512, qa_tokenizer.model_max_length)
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = qa_tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    version_2_with_negative = False
    n_best_size = 100
    max_answer_length = 15
    null_score_diff_threshold = 0.0

    answer_column_name = "answers"
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    #references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    #return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    return formatted_predictions

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=2)

    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_token_json, use_fast=True)
    #print(qa_tokenizer)

    qa_config = AutoConfig.from_pretrained(args.qa_config)
    qa_model = AutoModelForQuestionAnswering.from_config(qa_config).to("cuda")
    qa_model.load_state_dict(torch.load(args.qa_model))
    qa_model = accelerator.prepare(qa_model)
    #print(qa_model)

    test = json.load(open(args.test_file, 'r'))

    test_dataset = Dataset.from_list(test)
    
    predict_examples = test_dataset
    predict_dataset = predict_examples.map(
                qa_prepare_validation_features,
                batched=True,
                num_proc=24,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    data_collator = DataCollatorWithPadding(qa_tokenizer)
    predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
    predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.batch
        )
    #print(d)

    qa_model.eval()

    predict_dataloader = accelerator.prepare(predict_dataloader)
    all_start_logits = []
    all_end_logits = []
    pad_to_max_length = False

    for step, batch in enumerate(predict_dataloader):
        with torch.no_grad():
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())
    
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
    
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
    

    out = open(args.output, 'w')
    out.write("id,answer\n")
    for d in prediction:
        text = d['prediction_text'].replace('\"', '').strip()
        out.write(f"{d['id']},\"{text}\"\n")
    print("finish")

