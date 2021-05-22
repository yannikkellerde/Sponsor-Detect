import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes, pad_idx):
        super(BiLSTM_classifier,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.fc = nn.Linear(hidden_dim*2, num_classes)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, text_indices):
        embeds = self.word_embeddings(text_indices)
        lstm_out, _ = self.bilstm(embeds)
        class_space = self.fc(lstm_out)
        class_scores = self.softmax(class_space)
        return class_scores