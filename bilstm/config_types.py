# Automatically generated type hinting file for a .ini file

from typing import NamedTuple, List, Tuple

class Model(NamedTuple):
    embedding_dim:int
    hidden_dim:int
    num_layers:int

class Training(NamedTuple):
    lr:float
    number_epochs:int

class Data(NamedTuple):
    batch_size:int
    data_folder:str
    model_store_path:str
    model_name:str
    model_vocab_store:str

class Config(NamedTuple):
    Model:Model
    Training:Training
    Data:Data