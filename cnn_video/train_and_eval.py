from tqdm import tqdm
import torch

def evaluate(model, iterator_easy, iterator_hard, optimizer, criterion , metrics, device):
    model.eval()
    total_loss = torch.tensor([0.0])
    with torch.no_grad():
        for images, labels in tqdm(iterator_easy,desc="eval_ez"):
            images = images.to(device)
            labels = labels.to(device)
            predictions, pred2 = model(images)
            loss = criterion(predictions, labels)
            for key in metrics:
                metrics[key](pred2, labels)
            
            total_loss += loss.cpu()

        metric_total_ez = {}
        for key in metrics:
            metric_total_ez[key] = metrics[key].compute()
            metrics[key].reset()
        metric_total_ez["Loss"] = total_loss/len(iterator_easy)

        total_loss = torch.tensor([0.0])
        for images, labels in tqdm(iterator_hard,desc="eval_hard"):
            images = images.to(device)
            labels = labels.to(device)
            predictions, pred2 = model(images)

            loss = criterion(predictions, labels)

            for key in metrics:
                metrics[key](pred2, labels)
            
            total_loss += loss.cpu()

        metric_total_hard = {}
        for key in metrics:
            metric_total_hard[key] = metrics[key].compute()
            metrics[key].reset()
        metric_total_hard["Loss"] = total_loss/len(iterator_easy)

    return metric_total_ez, metric_total_hard


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