import argparse
import logging
from datasets import load_dataset
import os
from datetime import datetime
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main(config):
    os.environ["WANDB_WATCH"] = "false"
    if config.get("wandb_project", ""):
        os.environ["WANDB_PROJECT"] = config["wandb_project"]
    if config.get("do_train", True):
        train(config)
    else:
        test(config)

def train(config):
    logging.info(config)
    task_folder = f"{config.get('task_name', '')}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "train_config.yaml"), "w"))

    model_config = AutoConfig.from_pretrained(config.get("model", "distilbert-base-uncased"), num_labels=6)
    model = AutoModelForSequenceClassification.from_pretrained(config.get('model', "distilbert-base-uncased"),
                                                               config=model_config)
    train_config = config["train"]

    dataset = load_data(config)


def test(config, model=None):
    pass


def load_data(config):
    tokenizer = AutoTokenizer.from_pretrained(config.get("model", "distilbert-base-uncased"))
    base_path = "data/train_val_test/sponsor_nlp_data/"
    dataset = load_dataset("csv", delimiter="\t", quoting=3, column_names=['word', 'label'], data_files={
        "train": base_path + "train.tsv",
        "test": base_path + "test.tsv",
        "dev": base_path + "val.tsv"
    })

    def encode_batch(batch):
        text = batch['word']
        text = [str(i or '') for i in text]
        idx = (np.array([isinstance(t, str) for t in text]) - 1).nonzero()
        encoding = tokenizer(text, max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")
        return encoding

    dataset = dataset.map(encode_batch, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', "attention_mask", "token_type_ids", "label"])
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)

