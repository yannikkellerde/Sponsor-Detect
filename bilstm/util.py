from ast import literal_eval
import configparser
from collections import namedtuple
from typing import NamedTuple
from functools import reduce
from torch import nn

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