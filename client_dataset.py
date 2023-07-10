import torch
from torch.utils.data import Dataset


class ClientDataset(Dataset):
  def __init__(self, main_dataset, client_data_idxs):
    super().__init__()
    
    self.main_dataset = main_dataset
    self.client_data_idxs = client_data_idxs
    
  def __len__(self):
    return len(self.client_data_idxs)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
      
    return self.main_dataset[self.client_data_idxs[idx]]