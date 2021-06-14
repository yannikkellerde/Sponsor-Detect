from tqdm import trange,tqdm
from torchmetrics import Metric
from torchtext.legacy.datasets import SequenceTaggingDataset
from torchtext.legacy.data import Field,BucketIterator
from torchmetrics import F1,Accuracy
import torch
import argparse
import os,sys

from config_to_object import load_config
from bilstm.config_types import Config
from bilstm.util import pred_to_category,load_model, format_metrics
from bilstm.dataset import load_vocabs
from bilstm.model import Bidirectional_classifier

def evaluate(model, iterator, criterion, metrics):
    # SOURCE (MODIFIED): https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1%20-%20BiLSTM%20for%20PoS%20Tagging.ipynb

    total_loss = torch.tensor([0.0])
    
    model.eval()

    with torch.no_grad():    
        for batch in tqdm(iterator,desc="eval"):
            
            text = batch.text
            labels = batch.category

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

if __name__ == "__main__":
    HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Model loading stuff')
    parser.add_argument("--model_name","-m",type=str,help="Which torch model to load")
    parser.add_argument("--model_num","-n",type=int,help="Number (version) of the model to load")
    parser.add_argument("--dataset","-d",type=str,help="Which dataset to load")
    args = parser.parse_args()

    config:Config = load_config("config.ini")
    config:Config = load_config(os.path.join(HOME_PATH,config.Data.config_store,args.model_name+".ini"))

    text_dic,category_dic = load_vocabs(os.path.join(HOME_PATH,config.Data.model_vocab_store,args.model_name+".pkl"))
    text_field = Field(lower=True)
    category_field = Field(unk_token=None)
    fields = (("text",text_field),("category",category_field))

    test_data = SequenceTaggingDataset(args.dataset,fields)

    text_field.vocab = text_dic
    category_field.vocab = category_dic

    model = Bidirectional_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                                     config.Model.num_layers,len(text_dic),
                                     len(category_dic),text_dic.stoi["<pad>"],
                                     gru=config.Model.gru)

    model = model.to(DEVICE)

    metadata = load_model(os.path.join(HOME_PATH,config.Data.model_store_path,args.model_name,f"{args.model_name}_{args.model_num}.tar"),model)

    model.eval()

    print(metadata)

    test_iterator = BucketIterator(
            test_data,
            batch_size = config.Data.batch_size, device = DEVICE)

    metrics = {
        "Accuracy":Accuracy(ignore_index=category_field.vocab.stoi[category_field.pad_token]).to(DEVICE),
        "F1":F1(ignore_index=category_field.vocab.stoi[category_field.pad_token],num_classes=len(category_field.vocab),average=None).to(DEVICE)
    }

    if "weighting" in metadata:
        weighting = metadata["weighting"]
    else:
        #weighting = torch.Tensor([0.0000, 0.0527, 0.9473]).to(DEVICE)
        weighting = torch.Tensor([0.0000, 0.0027, 0.0343, 0.9630]).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=weighting)

    eval_metrics = evaluate(model,test_iterator,criterion,metrics)

    print(", ".join(f"Evaluation {key}: {value}" for key,value in format_metrics(eval_metrics,category_field).items()))