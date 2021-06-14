import torch
import torch.nn as nn

class BiLSTM_pic_classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.softmax = nn.Softmax(dim=2)
        

    def forward(self, x_padded, x_lens):
        packed_batch = nn.utils.rnn.pack_padded_sequence(x_padded, x_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed_batch)
        unpacked_batch, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        class_space = self.fc(unpacked_batch)
        class_scores = self.softmax(class_space)
        return class_scores
