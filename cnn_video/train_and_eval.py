from tqdm import tqdm
import torch

def evaluate(model, iterator, optimizer, criterion , metrics, device):
    model.eval()
    total_loss = torch.tensor([0.0])
    with torch.no_grad():
        for images, labels in tqdm(iterator,desc="eval"):
            images = images.to(device)
            labels = labels.to(device)
            predictions, pred2 = model(images)
            loss = criterion(predictions, labels)
            for key in metrics:
                metrics[key](pred2, labels)
            
            total_loss += loss.cpu()

        metric_total = {}
        for key in metrics:
            metric_total[key] = metrics[key].compute()
            metrics[key].reset()
        metric_total["Loss"] = total_loss/len(iterator)

    return metric_total


def train(model, iterator, optimizer, criterion, metrics, device):
    total_loss = torch.tensor([0.0])

    model.train()
    for images, labels in tqdm(iterator,desc="train"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions, pred2 = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        
        optimizer.step()
        for key in metrics:
            metrics[key](pred2,labels)
        
        total_loss += loss.cpu()

    metric_total = {}
    for key in metrics:
        metric_total[key] = metrics[key].compute()
        metrics[key].reset()
    metric_total["Loss"] = total_loss/len(iterator)

    return metric_total