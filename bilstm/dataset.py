from torchtext.legacy.data import BucketIterator, Field
from torchtext.legacy.datasets import SequenceTaggingDataset
import torch
import os
from bilstm.util import product
from itertools import combinations
HOME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..")

class DataHandler():
    def __init__(self,config,device,nrows=None):
        self.config = config
        self.device = device
        self.data_folder = os.path.join(HOME_PATH,self.config.data_folder)

        self.text_field = Field(lower=True)
        self.category_field = Field()
        fields = (("TEXT",self.text_field),("CATEGORY",self.category_field))
        self.train_data,self.val_data,self.test_data = SequenceTaggingDataset.splits(
            self.data_folder,train="train.tsv",validation="val.tsv",test="test.tsv",fields=fields)
        
        # Create Vocabulary and initialize from pretrained glove word embeddings
        self.text_field.build_vocab(self.train_data, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
        self.category_field.build_vocab(self.train_data)
        self.pad_idx = self.text_field.vocab.stoi[self.text_field.pad_token]

        # Build iterators that handle the batch size and batch examples of similar length together
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.val_data, self.test_data),
            batch_size = self.config.batch_size, device = device)

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

    def calc_category_weighting(self) -> torch.Tensor:
        weighting = torch.zeros(self.num_categories)
        denominator = sum(product(x) for x in combinations(self.category_appearance.values(),len(self.category_appearance)-1))
        for key in self.category_appearance:
            weighting[self.category_field.vocab.stoi[key]] = product(
                self.category_appearance[inkey] for inkey in self.category_appearance if inkey!=key) / denominator
        return weighting