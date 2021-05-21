import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange,tqdm
from bilstm.model import BiLSTM_classifier
from bilstm.util import load_config,lstm_weights_init
from bilstm.dataset import DataHandler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA not avaliable")

config = load_config("config.ini")

data_handler = DataHandler(config.Data,device)

model = BiLSTM_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                          data_handler.vocab_size,data_handler.num_categories,
                          data_handler.pad_idx)
lstm_weights_init(model)
optimizer = optim.Adam(model.parameters(),lr=config.Training.lr)

weighting = data_handler.calc_category_weighting()
loss_function = nn.CrossEntropyLoss(weight=weighting)