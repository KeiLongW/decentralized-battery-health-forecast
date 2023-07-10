import copy

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from client_dataset import ClientDataset
from constants import *
from masked_mae import MaskedMAELoss


class ClientTrainer():
  def __init__(self, args, model, device, main_dataset, data_idxs):
    self.args = args
    self.global_model = copy.deepcopy(model)
    self.model = model
    self.device = device
    self.main_dataset = main_dataset
    self.data_idxs = data_idxs
    self.train_loader = DataLoader(ClientDataset(self.main_dataset, self.data_idxs), 
                                   batch_size=self.args.client_batch_size, 
                                   shuffle=True)
    self.criterion = MaskedMAELoss()
    if self.args.client_optimizer == 'adam':      
      self.optim = Adam(self.model.parameters(), lr=self.args.client_learning_rate)
    elif self.args.client_optimizer == 'sgd':
      self.optim = SGD(self.model.parameters(), lr=self.args.client_learning_rate, momentum=self.args.client_sgd_optimizer_momentum)
    self.schedulear = StepLR(self.optim, step_size=self.args.step_lr_step_size, gamma=self.args.step_lr_gamma)
  
  def train_one_round(self):
    train_loss = 0
    
    for epoch in range(self.args.client_epochs):
      self.model.train()
      batch_loss = 0
      for x, x_feat, y, y_feat, x_lens, y_lens, *_ in self.train_loader:
        x = x.float().to(self.device)
        x_feat = x_feat.float().to(self.device)
        y = y.float().to(self.device)
        y_feat = y_feat.float().to(self.device)
        
        self.optim.zero_grad()
        pred = self.model(x, x_feat, y_feat, x_lens, y_lens)
        loss = self.criterion(pred, y)
        
        if self.args.fed_algorithm == 'FedProx':
          proximal_term = 0.
          for w, w_global in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_global).norm(2)
          loss += (self.args.fedprox_mu / 2) * proximal_term
        
        loss.backward()
        self.optim.step()
        
        batch_loss += loss.item()
        
      batch_loss /= len(self.train_loader)
      train_loss += batch_loss
      self.schedulear.step()
      
      print(f'Client epoch {epoch+1}/{self.args.client_epochs} training loss: {batch_loss:.4f}')
      
    train_loss /= self.args.client_epochs
    
    return self.model.state_dict() , train_loss   
    
    