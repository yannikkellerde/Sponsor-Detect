from bilstm.model import BiLSTM_classifier
import torch.optim as optim
import torch.nn as nn
from bilstm.util import load_config,get_ix_map,prepare_data,train_val_test_split
import pandas as pd
import numpy as np
from tqdm import trange,tqdm

config = load_config("config.ini")

data_df = pd.read_csv("../data/all_processed.csv",index_col=0,nrows=100000)
word_to_ix = get_ix_map(data_df,"word")
category_to_ix = get_ix_map(data_df,"category")
X,labels = prepare_data(data_df,word_to_ix,category_to_ix)

(X_train,y_train),(X_val,y_val),(X_test,y_test) = train_val_test_split(X,labels,config.Data.val_percent,config.Data.test_percent)

print(f"Dictionary size: {len(word_to_ix)}, Number of classes: {len(category_to_ix)}")
model = BiLSTM_classifier(config.Model.embedding_dim,config.Model.hidden_dim,len(word_to_ix),len(category_to_ix))
optimizer = optim.Adam(model.parameters(),lr=config.Training.lr)
loss_function = nn.NLLLoss()

for epoch in trange(300):  # again, normally you would NOT do 300 epochs, it is toy data
    losses = []
    for x, y in tqdm(zip(X_train,y_train)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.

        # Step 3. Run our forward pass.
        fx = model(x)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(fx, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Training loss: {np.mean(losses)}")