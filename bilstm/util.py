from ast import literal_eval
import configparser
from collections import namedtuple
import pandas as pd
import torch
import random
from typing import NamedTuple, Tuple

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

def get_ix_map(data:pd.DataFrame,column:str) -> dict:
    singles = data[column].unique()
    return dict(zip(singles,range(len(singles))))

def prepare_data(data:pd.DataFrame,word_to_ix:dict,category_to_ix:dict) -> tuple:
    inputs = []
    labels = []
    for vid_str in data["video"].unique():
        sentence_df = data[data["video"]==vid_str]
        inputs.append(torch.tensor([word_to_ix[x] for x in sentence_df["word"]]))
        labels.append(torch.tensor([category_to_ix[x] for x in sentence_df["category"]]))
    return inputs, labels

def train_val_test_split(X:list,labels:list,val_percent:int,test_percent:int) -> tuple:
    both = zip(X,labels)
    random.shuffle(both)
    new_X,new_labels = zip(*both)
    test_start = round(len(new_labels)-(test_percent*len(new_labels))/100)
    val_start = round(test_start-(val_percent*len(new_labels))/100)
    train = (new_X[:val_start],new_labels[:val_start])
    val = (new_X[val_start:test_start],new_labels[val_start:test_start])
    test = (new_X[test_start:],new_labels[test_start:])
    return train,val,test