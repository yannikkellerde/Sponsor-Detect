import json
import torch
from torch import nn
from datetime import datetime
import os


def save_model(model:nn.Module,optimizer:torch.optim.Optimizer,epoch:int, model_path:str,model_name:str):
    """method for storing deep learning models
    
    Args:
        model:      model that is to be saved
        optimizer:  optimizer with which the model was trained
        epoch:      epoch in which the model was saved
        model_path: path where the model should be saved
        model_name: name of the model
    """
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    save_dict = {
        "epoch":epoch,
        "timestamp":timestamp,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
    }
    torch.save(save_dict,os.path.join(model_path,f"{model_name}.tar"))