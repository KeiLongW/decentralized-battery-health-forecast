import os
import pickle
import random

import numpy as np
import torch
from scipy import stats
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MITStanfordFeaturesDataset(Dataset):
  def __init__(self, 
               data_dir_path,
               data_keys, 
               min_curve_len=100, 
               future_interval=1, 
               x_seq_len=2500, 
               y_seq_len=2500,
               downsample_ratio=1,
               input_ratio=None,
               curve_ratio=1,
               preprocessed_curve_ratio_idx=None,
               start_ratio=0):
    super().__init__()
    
    self.data_dir_path = data_dir_path
    self.data_keys = data_keys
    self.min_curve_len = min_curve_len
    self.future_interval = future_interval
    self.x_seq_len = x_seq_len
    self.y_seq_len = y_seq_len
    self.downsample_ratio = downsample_ratio
    self.input_ratio = input_ratio
    self.curve_ratio = curve_ratio
    self.preprocessed_curve_ratio_idx = preprocessed_curve_ratio_idx
    self.start_ratio = start_ratio
    
    self.data = pickle.load(open(os.path.join(self.data_dir_path, 'preprocessed_data.pkl'), 'rb'))
    self.load_data()
    
  def load_data(self):
    self.padded_x, self.padded_x_feat, self.padded_y, self.padded_y_feat, self.x_lens, self.y_lens, self.padded_time_deltas, self.battery_idxs, self.battery_stats, self.battery_data_sizes = self._prepare_seq_data()
    
  def _prepare_seq_data(self):
    x = []
    x_feat = []
    y = []
    y_feat = []
    time_deltas = []
    battery_idxs = {}
    battery_data_sizes = {}
    battery_idx_count = 0
    battery_stats = {}
    for key in self.data_keys:
      bat = self.data[key]
      if self.preprocessed_curve_ratio_idx is not None:
        curve_ratio = bat['curve_ratios'][self.preprocessed_curve_ratio_idx] if self.preprocessed_curve_ratio_idx < len(bat['curve_ratios']) else bat['curve_ratios'][-1]
      else:
        curve_ratio = self.curve_ratio
      caps = bat['capacities']
      caps = caps[:round(len(caps)*curve_ratio)]
      features = bat['features']
      features = features[:round(len(features)*curve_ratio)]
      
      if self.input_ratio is not None:        
        x, x_feat, y, y_feat, time_deltas = self._insert_sub_x_y(caps, features, round(len(caps)*self.input_ratio), x, x_feat, y, y_feat, time_deltas)
      else:
        for i in range(self.min_curve_len, len(caps)-self.min_curve_len, 1):          
          x, x_feat, y, y_feat, time_deltas = self._insert_sub_x_y(caps, features, i, x, x_feat, y, y_feat, time_deltas)
          
      idxs = list(range(battery_idx_count, len(x)))
      if len(idxs) > 0:
        battery_idxs[key] = idxs
        battery_data_sizes[key] = len(x) - battery_idx_count
        battery_idx_count = len(x)        
        battery_stats[key] = {
          'curve_ratio': curve_ratio,
          'curve_length': len(caps),
          'first_cap': caps[0].item(),
          'end_cap': caps[-1].item(),
          'cap_sd': np.std(np.array(caps)),
          'cap_slope': stats.linregress(np.arange(len(caps)), np.array(caps))[0],
          'idle_time': features[:, 0].sum(),
          'charge_time': features[:, 1].sum(),
          'discharge_time': features[:, 2].sum(),
        }
    
    x_lens = torch.tensor([i.shape[0] for i in x], dtype=torch.float32)
    y_lens = torch.tensor([i.shape[0] for i in y], dtype=torch.float32)
    padded_x = pad_sequence(x, batch_first=True)
    padded_x_feat = pad_sequence(x_feat, batch_first=True)
    padded_y = pad_sequence(y, batch_first=True)
    padded_y_feat = pad_sequence(y_feat, batch_first=True)
    padded_time_deltas = pad_sequence(time_deltas, batch_first=True)

    # align all sequences to the same length
    if padded_x.shape[1] < self.x_seq_len:
      padded_x = torch.cat((padded_x, torch.zeros((padded_x.shape[0], self.x_seq_len-padded_x.shape[1], 1))), dim=1)
      padded_x_feat = torch.cat((padded_x_feat, torch.zeros((padded_x_feat.shape[0], self.x_seq_len-padded_x_feat.shape[1], padded_x_feat.shape[2]))), dim=1)
    if padded_y.shape[1] < self.y_seq_len:
      padded_y = torch.cat((padded_y, torch.zeros((padded_y.shape[0], self.y_seq_len-padded_y.shape[1], 1))), dim=1)
      padded_y_feat = torch.cat((padded_y_feat, torch.zeros((padded_y_feat.shape[0], self.y_seq_len-padded_y_feat.shape[1], padded_y_feat.shape[2]))), dim=1)
    if padded_time_deltas.shape[1] < self.x_seq_len:
      padded_time_deltas = torch.cat((padded_time_deltas, torch.zeros((padded_time_deltas.shape[0], self.x_seq_len-padded_time_deltas.shape[1]))), dim=1)
    
    return padded_x, padded_x_feat, padded_y, padded_y_feat, x_lens, y_lens, padded_time_deltas, battery_idxs, battery_stats, battery_data_sizes
  
  def _insert_sub_x_y(self, caps, features, split_idx, x, x_feat, y, y_feat, time_deltas):
    
    # start from index 1 as the first capacity data is always 0 value
    start_idx = 1 if self.start_ratio == 0 else round(len(caps)*self.start_ratio)
    # randomly downsample to irregular sampling
    irr_idx = np.sort(random.sample(range(start_idx, split_idx), round((split_idx-start_idx)*self.downsample_ratio)))
    sub_x = caps[irr_idx].reshape(-1,1)
    # get features from input cycles
    sub_x_feat = features[irr_idx]

    sub_y = caps[split_idx::self.future_interval].reshape(-1,1)
    # get feature from output cycles
    sub_y_feat = features[split_idx::self.future_interval]
          
    x.append(sub_x)
    x_feat.append(sub_x_feat)
    y.append(sub_y)
    y_feat.append(sub_y_feat)
    # add 1 to the first element as the first element is the first time step
    time_deltas.append(torch.tensor(np.insert(np.diff(irr_idx), 0, 1, axis=0), dtype=torch.float32))
    
    return x, x_feat, y, y_feat, time_deltas
  
  def __len__(self):
    return len(self.padded_x)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    return self.padded_x[idx], self.padded_x_feat[idx], self.padded_y[idx], self.padded_y_feat[idx], self.x_lens[idx], self.y_lens[idx], self.padded_time_deltas[idx]