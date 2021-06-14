from tqdm import trange,tqdm
from torchmetrics import Metric
import torch
from bilstm.util import pred_to_category

def evaluate(model, iterator, optimizer, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    total_loss = torch.tensor([0.0])
    
    model.eval()

    
    for (x_padded, y_padded, x_lens, y_lens) in tqdm(iterator,desc="eval"):
        x_padded = x_padded.to("cuda")
        y_padded = y_padded.to("cuda")
        #x_lens = x_lens.to("cuda")
        
        optimizer.zero_grad()

        predictions = model(x_padded, x_lens)
        
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = y_padded.view(-1)
        

        loss = criterion(predictions, labels)
        
        for key in metrics:
            metrics[key](predictions, labels)
        
        total_loss += loss.cpu()

    metric_total = {}
    for key in metrics:
        metric_total[key] = metrics[key].compute()
        metrics[key].reset()
    metric_total["Loss"] = total_loss/len(iterator)

    return metric_total

def train(model, iterator, optimizer, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    total_loss = torch.tensor([0.0])

    model.train()
    
    for (x_padded, y_padded, x_lens, y_lens) in tqdm(iterator,desc="train"):
        x_padded = x_padded.to("cuda")
        y_padded = y_padded.to("cuda")

        optimizer.zero_grad()

        predictions = model(x_padded, x_lens)

        predictions = predictions.view(-1, predictions.shape[-1])
        labels = y_padded.view(-1)

        loss = criterion(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        for key in metrics:
            metrics[key](predictions,labels)
        
        total_loss += loss.cpu()

    metric_total = {}
    for key in metrics:
        metric_total[key] = metrics[key].compute()
        metrics[key].reset()
    metric_total["Loss"] = total_loss/len(iterator)

    return metric_total