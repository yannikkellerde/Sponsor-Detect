import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import video_image_dataset
from model import alex_net_like_cnn
import pandas as pd
import os
from torchmetrics import F1,Accuracy
from train_and_eval import train, evaluate
from utils import save_model
from tqdm import trange
import time

BATCH_SIZE = 128
LR = 1e-3
NUMBER_EPOCHS = 100

DATASET_PATH = "../data/image_dataset"
MODEL_STORE_PATH = "models"
MODEL_NAME = "cnn_model"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
val_df = pd.read_csv(os.path.join(DATASET_PATH, "val.csv"))
test_df = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))
val_hard_df = pd.read_csv(os.path.join(DATASET_PATH, "val_hard.csv"))

train_dataset = video_image_dataset(train_df, DATASET_PATH)
val_dataset = video_image_dataset(val_df, DATASET_PATH)
test_dataset = video_image_dataset(test_df, DATASET_PATH)
val_hard_dataset = video_image_dataset(val_hard_df, DATASET_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
val_hard_loader =  DataLoader(val_hard_dataset, batch_size=BATCH_SIZE)

model = alex_net_like_cnn(num_classes=2)
model.to(device)
optimizer = optim.Adam(model.parameters() ,lr=LR)

metrics = {
    "Accuracy":Accuracy().to(device),
    "F1":F1(num_classes=2, average=None).to(device)
}

loss_function = nn.CrossEntropyLoss().to(device)
eval_metrics_ez, eval_metrics_hard = evaluate(model, val_loader, val_hard_loader, optimizer,loss_function,metrics, device)
print("Epoch 0")
print("Easy Validation")
for key in eval_metrics_ez:
    print(f"Evaluation {key}:", eval_metrics_ez[key])
print("Hard Validation")
for key in eval_metrics_hard:
    print(f"Evaluation {key}:", eval_metrics_hard[key])

training_progress = []
best_model = 0
for epoch in trange(1, NUMBER_EPOCHS, desc="Epochs"):
    start_time = time.time()
    train_metrics = train(model, train_loader, optimizer, loss_function, metrics, device)
    eval_metrics_ez, eval_metrics_hard = evaluate(model, val_loader, val_hard_loader, optimizer, loss_function, metrics, device)
    if eval_metrics_hard["Accuracy"] > best_model:
        save_model(model, optimizer, epoch, MODEL_STORE_PATH, MODEL_NAME)
        best_model = eval_metrics_hard["Accuracy"]

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))

    progress = {
        "Epoch":epoch,
        "elapsed":elapsed,
        "train_loss":train_metrics["Loss"].item(),
        "train_accuracy":train_metrics["Accuracy"].item(),
        "valid_loss":eval_metrics_ez["Loss"].item(),
        "valid_accuracy":eval_metrics_ez["Accuracy"].item(),
    }
    training_progress.append(progress)

    print(f"\nEpoch {epoch}, epoch time: {elapsed}")
    for key in train_metrics:
        print(f"Training {key}:", train_metrics[key])
    print("Easy Validation")
    for key in eval_metrics_ez:
        print(f"Evaluation {key}:", eval_metrics_ez[key])
    print("Hard Validation")
    for key in eval_metrics_hard:
        print(f"Evaluation {key}:", eval_metrics_hard[key])
