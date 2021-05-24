from tqdm import trange,tqdm
from torchmetrics import Metric
import torch

def evaluate(model, iterator, optimizer, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    total_loss = torch.tensor([0.0])
    
    model.eval()
    
    for batch in tqdm(iterator,desc="eval"):
        
        text = batch.text
        labels = batch.category
        
        optimizer.zero_grad()

        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #labels = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #labels = [sent len * batch size]
        
        loss = criterion(predictions, labels)
        
        for key in metrics:
            metrics[key](predictions,labels)
        
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
    
    for batch in tqdm(iterator,desc="train"):
        
        text = batch.text
        orig_labels = batch.category
        
        optimizer.zero_grad()

        predictions = model(text)
        
        #predictions = [sent len, batch size, output dim]
        #labels = [sent len, batch size]
        
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = orig_labels.view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #labels = [sent len * batch size]

        if predictions.shape[0] != labels.shape[0]:
            print("\nShapes do not match",text.shape,orig_labels.shape,labels.shape,predictions.shape)
            print(text[:20,0],text[-2:,0],orig_labels[:2,0],orig_labels[-2:,0])
        
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