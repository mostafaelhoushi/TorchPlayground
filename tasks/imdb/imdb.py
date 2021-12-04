import argparse

import json
import os

import torch

# huggingface libraries
import transformers
import tasks.imdb.models as models
import datasets 

model_names = ["bert-base-cased"] # todo

normalize = "todo"

train_transforms = "todo"

validation_transforms = "todo"

raw_datasets = datasets.load_dataset("imdb") # todo: generalize this
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased") # todo: generalize this

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

preprocess = "todo" # todo validation_transforms


def train_dataset(data_dir): # todo: use data_dir
    return tokenized_datasets["train"]

def validation_dataset(data_dir): # todo: use data_dir
    return tokenized_datasets["test"]

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
    return len(batch["labels"])

# TODO: try the solution again in StackOverflow?
def forward(model, batch):
    return model(**batch)

def get_target(batch):
    return batch["labels"]

def default_loss_fn():
    return None

def get_loss(output, batch, loss_fn):
    del loss_fn # not using it
    return output.loss

def default_metrics_fn():
    return datasets.load_metric("accuracy")

def get_metrics(output, target, metrics_fn):
    logits = output.logits
    predictions = torch.argmax(logits, dim=-1)
    return list(metrics_fn.compute(predictions=predictions, references=target).values())

metrics = "todo"

idx2label = "todo"