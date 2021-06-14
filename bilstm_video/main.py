import torch.optim as optim
import torch.nn as nn
import torch
import os
import pickle
import time
from dataset import SequenzDataset
from model import BiLSTM_pic_classifier
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm, trange
from torchmetrics import F1,Accuracy
from itertools import combinations
from functools import reduce
from train import train,evaluate
from datetime import datetime

BATCH_SIZE = 8
HIDDEM_DIM = 256
NUM_LAYERS = 2
LR = 1e-3
NUMBER_EPOCHS = 100
MODEL_STORE_PATH = "data/models/"
MODEL_NAME = "video_bilstm2"
HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
CATEGORY_MAP = {
    "video": 1,
    "sponsor": 2,
}


def save_model(model:nn.Module,optimizer:torch.optim.Optimizer,epoch:int,train_metrics:dict,eval_metrics:dict,model_path:str,model_name:str):
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    save_dict = {
        "epoch":epoch,
        "timestamp":timestamp,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
    }
    for key in train_metrics:
        save_dict["train_"+key] = train_metrics[key]
    for key in eval_metrics:
        save_dict["eval_"+key] = eval_metrics[key]
    torch.save(save_dict,os.path.join(model_path,f"{model_name}_{epoch}.tar"))


def load_obj(file):
    """load an object

    name - name of the object to be loaded
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def lstm_weights_init(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)


def label_seq_to_number_tensor(label_seq):
    return [torch.tensor(CATEGORY_MAP[label]) for label in label_seq]


def padding_func(batch):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = torch.tensor(nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0))
    yy_pad = torch.tensor(nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0))
    return xx_pad, yy_pad, x_lens, y_lens

def product(iterable):
    return reduce(lambda x,y:x*y,iterable,1)

def clac_weighting(train_cats):
    train_cats = [item for seq in train_cats for item in seq]
    frequenzies = torch.zeros(len(CATEGORY_MAP))
    for train_cat in train_cats:
        frequenzies[train_cat-1] += 1
    print(frequenzies)
    weighting = torch.zeros(len(CATEGORY_MAP)+1)
    denominator = sum(product(x) for x in combinations(frequenzies,len(frequenzies)-1))
    for i in range(len(frequenzies)):
        weighting[i+1] = product(frequenzies[inkey] for inkey in range(len(frequenzies)) if inkey!=i)/denominator
    weighting = torch.tensor(weighting)
    print(weighting)
    return weighting


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not avaliable")

train_data = load_obj(os.path.join(HOME_PATH, "data/processed_videos/train.pkl"))
val_data = load_obj(os.path.join(HOME_PATH, "data/processed_videos/val.pkl"))
test_data = load_obj(os.path.join(HOME_PATH, "data/processed_videos/test.pkl"))

input_dim = len(train_data["embeddings"][0][0])
print(input_dim)

train_data_embeddings = [torch.tensor(embedding) for embedding in tqdm(train_data["embeddings"])]
train_data_cats = [torch.tensor(label_seq_to_number_tensor(label_seq)) for label_seq in tqdm(train_data['categorys'])] 
val_data_embeddings = [torch.tensor(embedding) for embedding in tqdm(val_data["embeddings"])]
val_data_cats = [torch.tensor(label_seq_to_number_tensor(label_seq)) for label_seq in tqdm(val_data['categorys'])]
test_data_embeddings = [torch.tensor(embedding) for embedding in tqdm(test_data["embeddings"])]
test_data_cats = [torch.tensor(label_seq_to_number_tensor(label_seq)) for label_seq in tqdm(test_data['categorys'])]

weighting = clac_weighting(train_data_cats)

train_dataset = SequenzDataset(train_data_embeddings, train_data_cats)
val_dataset = SequenzDataset(val_data_embeddings, val_data_cats)
test_dataset = SequenzDataset(test_data_embeddings, test_data_cats)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padding_func)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=padding_func)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=padding_func)

model = BiLSTM_pic_classifier(input_dim, HIDDEM_DIM, NUM_LAYERS, len(CATEGORY_MAP)+1)
lstm_weights_init(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters() ,lr=LR)

metrics = {
    "Accuracy":Accuracy(ignore_index=0).to(device),
    "F1":F1(ignore_index=0,num_classes=len(CATEGORY_MAP)+1,average=None).to(device)
}

loss_function = nn.CrossEntropyLoss(weight=weighting, ignore_index=0).to(device)
eval_metrics = evaluate(model, val_loader,optimizer,loss_function,metrics)
print("Epoch 0")
print(eval_metrics)
for key in eval_metrics:
    print(f"Evaluation {key}:", eval_metrics[key])
training_progress = []
for epoch in trange(1, NUMBER_EPOCHS, desc="Epochs"):
    start_time = time.time()
    train_metrics = train(model, train_loader,optimizer,loss_function,metrics)
    eval_metrics = evaluate(model, val_loader,optimizer,loss_function,metrics)

    save_model(model,optimizer,epoch,train_metrics,eval_metrics,os.path.join(HOME_PATH,MODEL_STORE_PATH), MODEL_NAME)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))

    progress = {
        "Epoch":epoch,
        "elapsed":elapsed,
        "train_loss":train_metrics["Loss"].item(),
        "train_accuracy":train_metrics["Accuracy"].item(),
        "valid_loss":eval_metrics["Loss"].item(),
        "valid_accuracy":eval_metrics["Accuracy"].item(),
    }
    training_progress.append(progress)

    print(f"\nEpoch {epoch}, epoch time: {elapsed}")
    for key in train_metrics:
        print(f"Training {key}:", train_metrics[key])
    for key in eval_metrics:
        print(f"Evaluation {key}:", eval_metrics[key])