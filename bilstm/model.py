import torch
import torch.nn as nn
import torch.nn.functional as F

class Bidirectional_classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, num_classes, pad_idx, dropout=0, gru=False):
        super(Bidirectional_classifier,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.droprate = dropout

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if gru:
            self.recurent = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=self.droprate)
        else:
            self.recurent = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=self.droprate)
        
        if self.droprate > 0:
            self.dropout = nn.Dropout(self.droprate)

        self.fc = nn.Linear(hidden_dim*2, num_classes)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, text_indices):
        embeds = self.word_embeddings(text_indices)
        lstm_out, _ = self.recurent(embeds)
        if self.droprate > 0:
            lstm_out = self.dropout(lstm_out)
        class_space = self.fc(lstm_out)
        class_scores = self.softmax(class_space)
        return class_scores