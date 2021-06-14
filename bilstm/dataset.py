from torchtext.legacy.data import BucketIterator, Field
from torchtext.legacy.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab

import torch
import os
from bilstm.config_types import Data
from bilstm.util import product
from itertools import combinations
import pickle
from typing import Tuple

class DataHandler():
    def __init__(self,config,device,home_path):
        self.config:Data = config
        self.device = device
        self.home_path = home_path
        self.vocab_path = os.path.join(self.home_path,self.config.model_vocab_store)
        self.data_folder = os.path.join(home_path,self.config.data_folder)

        self.text_field = Field(lower=True)
        self.category_field = Field(unk_token=None)
        fields = (("text",self.text_field),("category",self.category_field))
        self.train_data,self.val_data,self.test_data = SequenceTaggingDataset.splits(
            self.data_folder,train="train.tsv",validation="val.tsv",test="test.tsv",fields=fields)
        
        # Create Vocabulary and initialize from pretrained glove word embeddings
        self.text_field.build_vocab(self.train_data, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
        self.category_field.build_vocab(self.train_data)
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]
        self.category_pad_idx = self.category_field.vocab.stoi[self.category_field.pad_token]

        # Build iterators that handle the batch size and batch examples of similar length together
        self.train_iterator, self.val_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data),
            batch_size = self.config.batch_size, device = self.device)
            

    @property
    def num_categories(self):
        return len(self.category_field.vocab)

    @property
    def vocab_size(self):
        return len(self.text_field.vocab)

    @property
    def category_appearance(self) -> dict:
        total_examples = sum(self.category_field.vocab.freqs.values())
        return {key:self.category_field.vocab.freqs[key]/total_examples for key in self.category_field.vocab.freqs}

    def store_vocabs(self,fname):
        with open(os.path.join(self.vocab_path,fname),"wb") as f:
            pickle.dump({
                "text":self.text_field.vocab,
                "category":self.category_field.vocab
            },f)

    def calc_category_weighting(self) -> torch.Tensor:
        """Computes optimal weighting so that the prediction of each category will be equally important
        """
        weighting = torch.zeros(self.num_categories)
        denominator = sum(product(x) for x in combinations(self.category_appearance.values(),len(self.category_appearance)-1))
        for key in self.category_appearance:
            weighting[self.category_field.vocab.stoi[key]] = product(
                self.category_appearance[inkey] for inkey in self.category_appearance if inkey!=key) / denominator
        return weighting.to(self.device)

def load_vocabs(fname) -> Tuple[Vocab]:
    with open(fname,"rb") as f:
        dic = pickle.load(f)
    return dic["text"],dic["category"]