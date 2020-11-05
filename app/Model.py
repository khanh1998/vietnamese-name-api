import torch
from torch.nn import LSTM, Linear, LogSoftmax
import torch.nn as nn

class LSTM_classifier(nn.Module):
  def __init__(self, input_size=189, hidden_size=64, num_lstm_layer=2, dropout=0.9):
    super(LSTM_classifier, self).__init__();
    self.lstm = LSTM(
                  input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_lstm_layer,
                  batch_first=True,
                  dropout=dropout,
                  bidirectional=True
                  )
    self.linear = Linear(in_features=hidden_size*2, out_features=2)
    self.softmax = LogSoftmax(dim=1)
  def forward(self, x):
    '''
    x has shape (batch, length of name, feature)
    (16, 25, 189)
    '''
    # output of shape (batch, seq_len,num_directions * hidden_size)
    # (16, 25, 2 * 64)
    # h_n of shape (num_layers * num_directions, batch, hidden_size)
    # (2*2, 16, 64)
    # c_n of shape (num_layers * num_directions, batch, hidden_size)
    # (2*2, 16, 64)
    output, (h_n, c_n) = self.lstm(x)
    # print(f'output {output.shape}')
    # output_sum of shape (16, 128)
    output_sum = torch.sum(output, dim=1)
    # print(f'output sum {output_sum.shape}')
    # output1 of shape (16, 2)
    output1 = self.linear(output_sum)
    # print(f'output1 {output1.shape}')
    output2 = self.softmax(output1)
    # print(f'output2 {output2.shape}')
    return output2