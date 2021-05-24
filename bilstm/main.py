import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange,tqdm
import time
import os
from bilstm.model import BiLSTM_classifier
from bilstm.util import load_config,lstm_weights_init, load_model, save_model,format_metrics
from bilstm.dataset import DataHandler
from bilstm.train import train,evaluate
from torchmetrics import F1,Accuracy

HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not avaliable")

config = load_config("config.ini")

data_handler = DataHandler(config.Data,device,HOME_PATH)

model = BiLSTM_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                          config.Model.num_layers,data_handler.vocab_size,
                          data_handler.num_categories,data_handler.pad_idx)
lstm_weights_init(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=config.Training.lr)

weighting = data_handler.calc_category_weighting()
loss_function = nn.CrossEntropyLoss(weight=weighting)

metrics = {
    "Accuracy":Accuracy(ignore_index=data_handler.category_pad_idx).to(device),
    "F1":F1(ignore_index=data_handler.category_pad_idx,num_classes=data_handler.num_categories,average=None).to(device)
}

eval_metrics = evaluate(model,data_handler.val_iterator,optimizer,loss_function,metrics)
print("Epoch 0")
print(", ".join(f"Evaluation {key}: {value}" for key,value in format_metrics(eval_metrics,data_handler).items()))

for epoch in trange(1,config.Training.number_epochs+1,desc="Epochs"):
    start_time = time.time()
    train_metrics = train(model,data_handler.train_iterator,optimizer,loss_function,metrics)
    eval_metrics = evaluate(model,data_handler.val_iterator,optimizer,loss_function,metrics)

    save_model(model,optimizer,epoch,train_metrics,eval_metrics,os.path.join(HOME_PATH,config.Data.model_store_folder,"bilstm"))

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
    print(f"\nEpoch {epoch}, epoch time: {elapsed}")
    print(", ".join(f"Training {key}: {value}" for key,value in format_metrics(train_metrics,data_handler).items()))
    print(", ".join(f"Evaluation {key}: {value}" for key,value in format_metrics(eval_metrics,data_handler).items()))