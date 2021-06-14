import argparse
from pickle import load
from config_to_object import load_config
import torch.optim as optim
import torch.nn as nn
import torch
import os
from torchtext.legacy.datasets import SequenceTaggingDataset
from torchtext.legacy.data import Field
from collections import Counter

from frontend.inference_to_html import to_html
from bilstm.config_types import Config
from bilstm.model import Bidirectional_classifier
from bilstm.util import load_model
from bilstm.dataset import load_vocabs
from bilstm.train import get_prediction_combo

HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")

parser = argparse.ArgumentParser(description='Inference options')
parser.add_argument("--model_name","-m",type=str,help="Which torch model to load")
parser.add_argument("--model_num","-n",type=int,help="Number (version) of the model to load")
parser.add_argument("--input","-i",type=str,help="location of input file")
parser.add_argument("--output","-o",type=str,help="output file location")
args = parser.parse_args()

if args.model_name is None or args.model_num is None or args.input is None:
    print("You need to provide a model name as well as a model number and an input file")

config:Config = load_config("config.ini")
#config:Config = load_config(os.path.join(HOME_PATH,config.Data.config_store,args.model_name+".ini"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_dic,category_dic = load_vocabs(os.path.join(HOME_PATH,config.Data.model_vocab_store,args.model_name+".pkl"))
text_field = Field(lower=True)
category_field = Field(unk_token=None)
fields = (("text",text_field),("category",category_field))

test_data = SequenceTaggingDataset(args.input,fields)

text_field.vocab = text_dic
category_field.vocab = category_dic

model = Bidirectional_classifier(config.Model.embedding_dim,config.Model.hidden_dim,
                          config.Model.num_layers,len(text_dic),
                          len(category_dic),text_dic.stoi["<pad>"],
                          gru=config.Model.gru)

model = model.to(DEVICE)

metadata = load_model(os.path.join(HOME_PATH,config.Data.model_store_path,args.model_name,f"{args.model_name}_{args.model_num}.tar"),model)
model.eval()

for i in range(7,8):
    annotated = get_prediction_combo(model,text_dic,category_dic,test_data.examples[i],device=DEVICE)

    text,preds = zip(*annotated)
    to_html(text,test_data.examples[i].category,preds,os.path.abspath(os.path.join(HOME_PATH,config.Frontend.results_location,"test_6.html")),
            os.path.join(HOME_PATH,config.Frontend.template_location),os.path.join(HOME_PATH,config.Frontend.css_location))

    with open("prediction_examples.txt","a") as f:
        f.write("\n\n\n")
        f.write(str(annotated))