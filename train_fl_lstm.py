import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchinfo import summary

from arg_parser import parse_args
from constants import *
from server_trainer import ServerTrainer


class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, lstm_layers=1, bidirectional=True):
    super().__init__()
    
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.lstm_layers = lstm_layers
    self.bidirectional = bidirectional
    
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.lstm_layers, bidirectional=self.bidirectional)

  def forward(self, x, x_lens):
    self.lstm.flatten_parameters()
    
    x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
    x, (h, c) = self.lstm(x)
    y, _ = pad_packed_sequence(x, batch_first=True)
    return y, (h, c)
    
class Decoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, lstm_layers=1, bidirectional=True):
    super().__init__()
    
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.lstm_layers = lstm_layers
    self.bidirectional = bidirectional
    
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.lstm_layers, bidirectional=self.bidirectional)
    self.linear = nn.Linear(hidden_dim * 2 if self.bidirectional else hidden_dim, 1)

  def forward(self, x, h, c, x_lens):
    self.lstm.flatten_parameters()
    
    seq_len = x.shape[1]
    
    x = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
    x, (h, c) = self.lstm(x, (h, c))
    x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_len)
    y = self.linear(x)    
    return y
  
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, x, x_feat, y_feat, x_lens, y_lens):
    _, (h, c) = self.encoder(torch.cat((x, x_feat), dim=-1), x_lens)
    y = self.decoder(y_feat, h, c, y_lens)
    return y

class TrainFLLSTM(ServerTrainer):
  def __init__(self, args):
    super().__init__(args)
    
    print(self.args.exp_name)
    print(self.args)
    
    self.encoder = Encoder(self.encoder_input_dim, self.args.encoder_hidden_dim, self.args.lstm_layers, self.args.lstm_bidirectional).to(self.device)
    self.decoder = Decoder(self.decoder_input_dim, self.args.decoder_hidden_dim, self.args.lstm_layers, self.args.lstm_bidirectional).to(self.device)
    self.model = DataParallel(Seq2Seq(self.encoder, self.decoder), device_ids=self.device_ids).to(self.device)
    
    summary(self.model)
    
def main():
  args = parse_args()
  trainer = TrainFLLSTM(args)
  
  if (args.testing):
    trainer.load_model()
    trainer.test()
  elif (args.testing_late_start):
    trainer.load_model()
    trainer.test_late_start()
  else:      
    if (args.train_baseline):
      trainer.train_baseline()
    else:
      trainer.train()
    trainer.save_model()
    trainer.test()
  
if __name__ == '__main__':
  main()