import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AdamW, AutoConfig


class TransformerClassifier:
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model_config = AutoConfig.from_pretrained(config.get("model", "distilbert-base-uncased"), num_labels=6)
        self.transformer = AutoModelForTokenClassification.from_pretrained(config['model'], config=model_config).to(self.device)
        self.transformer.train()
        self.optim = AdamW(self.transformer.parameters(), lr=config['train'].get('learning_rate', 5e-5))

    def train(self, config, dataset):
        train_loader = DataLoader(dataset['train'], batch_size=config['train_batchsize'], drop_last=True)
        for epoch in range(config['epochs']):
            for batch in train_loader:
                self.optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                self.optim.step()

        def eval(eval_dataset):
            self.transformer.eval()