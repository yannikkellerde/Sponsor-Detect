from tqdm import trange,tqdm
from torchmetrics import Metric
import torch
from bilstm.util import pred_to_category

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

def predict(model,vocab,text,device):
    in_tensor = torch.tensor([vocab.stoi[word] if word in vocab.stoi else vocab.stoi["<unk>"] for word in text],device=device)
    in_tensor = in_tensor.view(-1,1)
    pred = model(in_tensor)
    return pred[:,0]

def get_prediction_combo(model,text_vocab,category_vocab,example,device):
    preds = predict(model,text_vocab,example.text,device)
    categories = pred_to_category(preds,category_vocab)
    correct = [a==b for a,b in zip(categories,example.category)]
    out_list = list(zip(example.text,categories))
    print("Accuracy",sum(correct)/len(correct))
    return out_list
