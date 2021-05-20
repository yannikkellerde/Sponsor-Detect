import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(BiLSTM_classifier,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden2class = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, text_indices):
        embeds = self.word_embeddings(text_indices)
        lstm_out, _ = self.bilstm(embeds.view(len(text_indices), 1, self.hidden_dim))
        tag_space = self.hidden2class(lstm_out.view(len(text_indices), self.hidden_dim*2))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores