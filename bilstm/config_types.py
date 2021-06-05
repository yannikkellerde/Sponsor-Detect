# Automatically generated type hinting file for a .ini file
# Generated with config-to-object https://pypi.org/project/config-to-object/1.0.0/
# Run "ini_typefile your_config.ini type_file.py" to create a new type file

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
    config_store:str
    progress_store:str

class Frontend(NamedTuple):
    template_location:str
    css_location:str
    results_location:str

class Config(NamedTuple):
    Model:Model
    Training:Training
    Data:Data
    Frontend:Frontend