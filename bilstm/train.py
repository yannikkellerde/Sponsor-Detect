from tqdm import trange,tqdm
from torchmetrics import Metric

def evaluate(model, iterator, optimizer, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    metric_total = {key:0 for key in metrics}
    metric_total["Loss"] = 0
    
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
            metric_total[key] += metrics[key](predictions,labels).item()
        
        metric_total["Loss"] += loss.item()

    for key in metric_total:
        metric_total[key] /= len(iterator)
        
    return metric_total

def train(model, iterator, optimizer, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    metric_total = {key:0 for key in metrics}
    metric_total["Loss"] = 0

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
        
        loss = criterion(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        metric_total["Loss"] += loss.item()
        for key in metrics:
            metric_total[key] += metrics[key](predictions,labels).item()
        
    for key in metric_total:
        metric_total[key] /= len(iterator)

    return metric_total