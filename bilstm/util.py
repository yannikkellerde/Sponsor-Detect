from ast import literal_eval
import configparser
from collections import namedtuple
from typing import NamedTuple
from functools import reduce
from torch import nn
import torch
import numpy as np
from datetime import datetime

def multidict_to_namedtuple(dic:dict,name:str) -> NamedTuple:
    for key in dic:
        if type(dic[key]) == dict:
            dic[key] = multidict_to_namedtuple(dic[key],key)
    return namedtuple(name,dic.keys())(*dic.values())

def load_config(filename:str) -> NamedTuple:
    config_obj = configparser.ConfigParser(inline_comment_prefixes=";")
    config_obj.read(filename)
    config_dict = config_obj._sections
    for key in config_dict:
        for key2 in config_dict[key]:
            try:
                config_dict[key][key2] = literal_eval(config_dict[key][key2])
            except:
                pass
    return multidict_to_namedtuple(config_dict,"config")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def product(iterable):
    return reduce(lambda x,y:x*y,iterable,1)

def lstm_weights_init(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

def sentence_from_indices(indices,vocab):
    return [vocab.itos[i] for i in indices]

def save_model(model:nn.Module,optimizer:torch.optim.Optimizer,epoch:int,train_metrics:dict,eval_metrics:dict,filepath_base:str):
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
    torch.save(save_dict,filepath_base+f"_{epoch}_{timestamp}.tar")

def load_model(filepath:str,model:nn.Module,optimizer:torch.optim.Optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint["model_state_dict"]
    del checkpoint['optimizer_state_dict']
    return checkpoint

def format_metrics(metrics,data_handler):
    out_metrics = {}
    out_metrics["F1"] = {data_handler.category_field.vocab.itos[i]:(value.item() if np.isnan(value.item()) else round(value.item(),4)) for i,value in enumerate(metrics["F1"])}
    for key,value in metrics.items():
        if key!="F1":
            if type(value) == torch.Tensor and len(value.shape)==1 and value.shape[0]==1:
                out_metrics[key] = round(value.item(),4)
            else:
                out_metrics[key] = value
    return out_metrics