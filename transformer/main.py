import argparse
import logging
from datasets import Dataset, DatasetDict, load_metric
import os
from datetime import datetime
import yaml
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
import torch

#from sklearn.metrics import f1_score
#from torchmetrics import F1, Accuracy


# TODO: to categorical

from transformer.model import TransformerClassifier

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
        evaluate(config)


def train(config):
    logging.info(config)
    task_folder = f"{config.get('task_name', '')}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "train_config.yaml"), "w"))

    model_config = AutoConfig.from_pretrained(config.get("model", "distilbert-base-uncased"), num_labels=6)
    model = AutoModelForTokenClassification.from_pretrained(config.get('model', "distilbert-base-uncased"),
                                                               config=model_config)
    train_config = config["train"]


    #model = TransformerClassifier(config)

    global label_list
    global metric


    dataset, collator, tokenizer, label_list = load_data(config)

    metric = load_metric("seqeval")
    #metric = F1(num_classes=6, average=None)



    #model.train(train_config, dataset)

    training_args = TrainingArguments(
        learning_rate=train_config.get("learning_rate", 5e-5),
        num_train_epochs=train_config["epochs"],
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("train_batchsize", 16),
        per_device_eval_batch_size=train_config.get("eval_batchsize", 32),
        logging_steps=train_config.get("logging_steps", 10),
        output_dir=output_dir,
        overwrite_output_dir=True,
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        fp16=train_config.get("amp", True),
        eval_steps=train_config.get("eval_steps", 250),
        evaluation_strategy="steps",
        #load_best_model_at_end=True,
        #metric_for_best_model="accuracy",
        #save_total_limit=train_config.get("save_total_limit", None),
        run_name=task_folder,
        report_to=config.get("report_to", "all"),
        skip_memory_metrics=config.get("skip_memory_metrics", True)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['dev'],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        #do_save_full_model=True
    )
    trainer.train()
    # TODO: Saving models

def compute_metrics(data):
    prediction_probs, labels = data
    predictions = np.argmax(prediction_probs, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    #results = f1_score(true_labels, true_predictions, average='samples')
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


def evaluate(config, model=None):
    pass


def load_data(config):

    tokenizer = AutoTokenizer.from_pretrained(config.get("model", "distilbert-base-uncased"))
    base_path = "data/train_val_test/sponsor_nlp_data/"

    df_train = pd.read_csv(base_path + "train.tsv", sep='\t', names=['word', 'label'])
    df_dev = pd.read_csv(base_path + "val.tsv", sep='\t', names=['word', 'label'])
    df_test = pd.read_csv(base_path + "test.tsv", sep='\t', names=['word', 'label'])

    label_list = df_train['label'].unique()
    label_dict = {}
    for i, label in enumerate(label_list):
        label_dict[label] = i

    if config['debug']:
        df_train = df_train[:500000]
        df_dev = df_dev[:50000]
        df_test = df_test[:50000]
    print()

    # TODO: Very ineffective
    def group_in_chunks(df, chunk_size):

        i = 0
        grouped_data = []
        while i < df.shape[0]:
            chunk = df[i:i+chunk_size]
            grouped_data.append([list(chunk['word']), list(chunk['label'])])
            i += chunk_size
        return pd.DataFrame(grouped_data, columns=['word', 'labels'])

    for df in [df_train, df_dev, df_test]:
        df['label'] = df['label'].apply(lambda x: label_dict.get(x))

    df_train = group_in_chunks(df_train, int(config['max_seq_len'] * 3/4))
    df_dev = group_in_chunks(df_dev, int(config['max_seq_len'] * 3/4))
    df_test = group_in_chunks(df_test, int(config['max_seq_len'] * 3/4))
    print()

    train_data = Dataset.from_pandas(df_train)
    dev_data = Dataset.from_pandas(df_dev)
    test_data = Dataset.from_pandas(df_test)

    dataset = DatasetDict()
    dataset['train'] = train_data
    dataset['dev'] = dev_data
    dataset['test'] = test_data

    #dataset = load_dataset("csv", delimiter="\t", quoting=3, column_names=['word', 'label'], data_files={
    #    "train": base_path + "train.tsv",
    #    "test": base_path + "test.tsv",
    #    "dev": base_path + "val.tsv"
    #})
    print()
    #if config['debug']:
    #    dataset['train'] = Dataset.from_dict(dataset['train'][:50000])
    #    dataset['dev'] = Dataset.from_dict(dataset['dev'][:5000])
    #    dataset['test'] = Dataset.from_dict(dataset['test'][:5000])


    def encode_batch(batch):
        text = batch['word']
        #text = [str(i or '') for i in text]
        idx = (np.array([isinstance(t, str) for t in text]) - 1).nonzero()
        encoding = tokenizer(text, truncation=True, is_split_into_words=True)
        labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for id in word_ids:
            if id is None:
                labels.append(-100)
            elif id != previous_word_idx:
                labels.append(batch['labels'][id])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
            else:
                labels.append(-100)
            previous_word_idx = id

        encoding['labels'] = labels
        return encoding

    dataset = dataset.map(encode_batch, batched=False)
    #dataset.set_format(type="torch", columns=['input_ids', "attention_mask", "label"])
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return dataset, data_collator, tokenizer, label_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)
