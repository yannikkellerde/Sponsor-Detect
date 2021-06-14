import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange,tqdm
import time
import os
import pandas as pd
from shutil import copyfile
import argparse

from config_to_object import load_config
from bilstm.model import Bidirectional_classifier
from bilstm.util import lstm_weights_init, load_model, save_model,format_metrics,pred_to_category
from bilstm.dataset import DataHandler
from bilstm.train import train,get_prediction_combo
from bilstm.evaluate import evaluate
from torchmetrics import F1,Accuracy
from bilstm.config_types import Config

parser = argparse.ArgumentParser(description='Model loading stuff')
parser.add_argument("--model_name","-m",type=str,default=None,help="Which torch model to load")
args = parser.parse_args()


HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not avaliable")

config:Config = load_config("config.ini")
copyfile("config.ini",os.path.join(HOME_PATH,config.Data.config_store,config.Data.model_name+".ini"))

data_handler = DataHandler(config.Data,device,HOME_PATH)
data_handler.store_vocabs(os.path.join(HOME_PATH,config.Data.model_vocab_store,config.Data.model_name+".pkl"))

model = Bidirectional_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                                 config.Model.num_layers,data_handler.vocab_size,
                                 data_handler.num_categories,data_handler.pad_idx,
                                 gru=config.Model.gru)
lstm_weights_init(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=config.Training.lr)

if args.model_name is not None:
    metadata = load_model(os.path.join(HOME_PATH,config.Data.model_store_path,args.model_name),model,optimizer=optimizer)
    print(metadata)

weighting = data_handler.calc_category_weighting()
print(weighting)
loss_function = nn.CrossEntropyLoss(weight=weighting)

metrics = {
    "Accuracy":Accuracy(ignore_index=data_handler.category_pad_idx).to(device),
    "F1":F1(ignore_index=data_handler.category_pad_idx,num_classes=data_handler.num_categories,average=None).to(device)
}

eval_metrics = evaluate(model,data_handler.val_iterator,loss_function,metrics)
print("Epoch 0")
print(", ".join(f"Evaluation {key}: {value}" for key,value in format_metrics(eval_metrics,data_handler.category_field).items()))

training_progress = []

for epoch in trange(1,config.Training.number_epochs+1,desc="Epochs"):
    start_time = time.time()
    train_metrics = train(model,data_handler.train_iterator,optimizer,loss_function,metrics)
    eval_metrics = evaluate(model,data_handler.val_iterator,loss_function,metrics)

    save_model(model,optimizer,epoch,train_metrics,eval_metrics,weighting,os.path.join(HOME_PATH,config.Data.model_store_path),config.Data.model_name)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))

    progress = {
        "Epoch":epoch,
        "elapsed":elapsed,
        "train_loss":train_metrics["Loss"].item(),
        "train_accuracy":train_metrics["Accuracy"].item(),
        "valid_loss":eval_metrics["Loss"].item(),
        "valid_accuracy":eval_metrics["Accuracy"].item()
    }
    for i,value in enumerate(train_metrics["F1"]):
        progress[f"train_F1_{data_handler.category_field.vocab.itos[i]}"] = value.item()
    for i,value in enumerate(eval_metrics["F1"]):
        progress[f"valid_F1_{data_handler.category_field.vocab.itos[i]}"] = value.item()
    training_progress.append(progress)
    pd.DataFrame(training_progress).to_csv(os.path.join(HOME_PATH,config.Data.progress_store,config.Data.model_name+".csv"))

    prediction = get_prediction_combo(model,data_handler.text_field.vocab,data_handler.category_field.vocab,data_handler.test_data.examples[epoch],device)

    with open("predictions.txt","a") as f:
        f.write("\n\n\n")
        f.write(str(prediction))

    print(f"\nEpoch {epoch}, epoch time: {elapsed}")
    print(", ".join(f"Training {key}: {value}" for key,value in format_metrics(train_metrics,data_handler.category_field).items()))
    print(", ".join(f"Evaluation {key}: {value}" for key,value in format_metrics(eval_metrics,data_handler.category_field).items()))