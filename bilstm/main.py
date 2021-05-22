import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange,tqdm
import time
from bilstm.model import BiLSTM_classifier
from bilstm.util import load_config,lstm_weights_init
from bilstm.dataset import DataHandler
from bilstm.train import train,evaluate
from torchmetrics import F1,Accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not avaliable")

config = load_config("config.ini")

data_handler = DataHandler(config.Data,device)

model = BiLSTM_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                          data_handler.vocab_size,data_handler.num_categories,
                          data_handler.pad_idx)
lstm_weights_init(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=config.Training.lr)

weighting = data_handler.calc_category_weighting()
loss_function = nn.CrossEntropyLoss(weight=weighting)

metrics = {
    "Accuracy":Accuracy(ignore_index=data_handler.category_pad_idx).to(device),
    "F1":F1(ignore_index=data_handler.category_pad_idx,num_classes=data_handler.num_categories).to(device)
}

for i in trange(config.Training.number_epochs,desc="Epochs"):
    start_time = time.time()
    train_metrics = train(model,data_handler.train_iterator,optimizer,loss_function,metrics)
    eval_metrics = evaluate(model,data_handler.val_iterator,optimizer,loss_function,metrics)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f"Epoch {i+1}, epoch time: {elapsed}")
    print(", ".join(f"Training {key}: {round(value,4)}" for key,value in train_metrics.items()))
    print(", ".join(f"Evaluation {key}: {round(value,4)}" for key,value in eval_metrics.items()))