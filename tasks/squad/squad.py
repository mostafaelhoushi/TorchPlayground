import argparse

import json
import os

import torch

# huggingface libraries
import transformers
import tasks.squad.models as models
import datasets 

from tasks.squad.utils_qa import postprocess_qa_predictions

model_names = ["bert-base-cased"] # todo

normalize = "todo"

train_transforms = "todo"

validation_transforms = "todo"

def load_tokenized_datasets(data_dir):
    raw_dataset = datasets.load_dataset("squad", data_dir=data_dir) # todo: generalize this
    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased") # todo: generalize this

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = raw_dataset.map(preprocess, batched=True)
    #import pdb; pdb.set_trace();
    tokenized_dataset = tokenized_dataset.remove_columns(["answers", "context", "id", "question", "title"])
    #tokenized_dataset = tokenized_dataset.remove_columns(["context", "question", "title"])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def train_dataset(data_dir=None): # todo: use data_dir
    tokenized_dataset = load_tokenized_datasets(data_dir)
    return tokenized_dataset["train"]

def validation_dataset(data_dir=None): # todo: use data_dir
    tokenized_dataset = load_tokenized_datasets(data_dir)
    return tokenized_dataset["validation"]

def default_epochs():
    return 3

def default_initial_lr():
    return 5e-5

def default_lr_scheduler(optimizer, num_epochs, steps_per_epoch, start_epoch=0):
    # todo: update with start_epoch
    num_training_steps = num_epochs * steps_per_epoch 
    return transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

def default_optimizer(model, lr, momentum, weight_decay):
    return transformers.AdamW(model.parameters(), lr,
                              weight_decay=weight_decay)

def to_device(batch, device, gpu_id):
    # todo: deal with gpu_id
    return {k: v.to(device) for k, v in batch.items()}

def get_batch_size(batch):
    return len(batch["start_positions"])

# TODO: try the solution again in StackOverflow?
def forward(model, batch):
    return model(**batch)

def get_target(batch):
    return batch

def default_loss_fn():
    return None

def get_loss(output, batch, loss_fn):
    del loss_fn # not using it
    return output.loss

def default_metrics_fn():
    return datasets.load_metric("squad")

def get_metrics(output, target, metrics_fn):
    def post_processing_function(examples, features, predictions, stage="train"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        import pdb; pdb.set_trace();
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=(predictions.start_logits, predictions.end_logits),
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    batch = target # here target is just the batch
    eval_prediction = post_processing_function(batch, train_dataset(), output, stage="train")

    return list(metrics_fn.compute(predictions=eval_prediction.predictions, references=eval_prediction.label_ids).values())
 
metrics = "todo"

idx2label = "todo"
