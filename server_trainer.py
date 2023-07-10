import copy
import json
import os
import random
import time
from abc import ABC
from datetime import datetime, timedelta

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from client_trainer import ClientTrainer
from constants import *
from masked_mae import MaskedMAELoss
from masked_mape_loss import MaskedMAPELoss
from masked_rmse_loss import MaskedRMSELoss
from mit_stanford_features_dataset import MITStanfordFeaturesDataset


class ServerTrainer(ABC):
  def __init__(self, args):    
    self.start_timestamp = time.time()
    
    self.args = args
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.device_ids = list(range(torch.cuda.device_count()))
    
    self.train_set = MITStanfordFeaturesDataset(self.args.data_dir,
                                        data_keys=TRAIN_DATA_KEYS,
                                        min_curve_len=self.args.min_curve_len,
                                        future_interval=self.args.future_interval,
                                        downsample_ratio=self.args.downsample_ratio)
    self.val_set = MITStanfordFeaturesDataset(self.args.data_dir,
                                        data_keys=VAL_DATA_KEYS,
                                        min_curve_len=self.args.min_curve_len,
                                        future_interval=self.args.future_interval,
                                        downsample_ratio=self.args.downsample_ratio)
    # capacity (1) + num of features
    self.encoder_input_dim = 1 + self.train_set.padded_x_feat.shape[-1]
    self.decoder_input_dim = self.train_set.padded_y_feat.shape[-1]
    
    self.val_loader = DataLoader(self.val_set, batch_size=self.args.val_batch_size, shuffle=True)
    
    self.rmse = MaskedRMSELoss()
    self.mape = MaskedMAPELoss()
    self.mae = MaskedMAELoss()
    
  def train_baseline(self):
    train_loader = DataLoader(self.train_set, batch_size=self.args.baseline_batch_size, shuffle=True)
    optim = Adam(self.model.parameters(), lr=self.args.baseline_learning_rate)
    criterion = MaskedMAELoss()
    all_train_losses = []
    val_mae_losses = []
    val_mape_losses = []
    val_rmse_losses = []
    
    for epoch in range(self.args.baseline_epoch):
      self.model.train()
      train_losses = []
      pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}/{self.args.baseline_epoch}')
      for x, x_feat, y, y_feat, x_lens, y_lens, *_ in pbar:
        x = x.float().to(self.device)
        x_feat = x_feat.float().to(self.device)
        y = y.float().to(self.device)
        y_feat = y_feat.float().to(self.device)
        
        optim.zero_grad()
        pred = self.model(x, x_feat, y_feat, x_lens, y_lens)
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        
        train_losses.append(loss.item())
        pbar.set_postfix({"Training loss": sum(train_losses) / len(train_losses)})
      
      all_train_losses.append(train_losses)
      
      val_mae_loss, val_mape_loss, val_rmse_loss = self._do_eval()      
      print(f'Epoch {epoch+1}/{self.args.baseline_epoch}: Val MAE: {val_mae_loss:.4f} | Val MAPE: {val_mape_loss:.4f} | Val RMSE: {val_rmse_loss:.4f}')
      val_mae_losses.append(val_mae_loss)
      val_mape_losses.append(val_mape_loss)
      val_rmse_losses.append(val_rmse_loss)
      
    os.makedirs(f'results/{self.args.exp_name}', exist_ok=True)
    json.dump(all_train_losses, open(f'results/{self.args.exp_name}/all_train_losses.json', 'w'))
    json.dump(val_mae_losses, open(f'results/{self.args.exp_name}/val_mae_losses.json', 'w'))
    json.dump(val_mape_losses, open(f'results/{self.args.exp_name}/val_mape_losses.json', 'w'))
    json.dump(val_rmse_losses, open(f'results/{self.args.exp_name}/val_rmse_losses.json', 'w'))
    
  def train(self):
    starting_global_round = 0
    if (self.args.continue_training):
      snapshot_dir = f'results/{self.args.exp_name}/model_snapshot'
      latest_snapshot = max([int(f.split('.')[0]) for f in os.listdir(snapshot_dir)])
      self.model.load_state_dict(torch.load(f'{snapshot_dir}/{latest_snapshot}.pth'))
      starting_global_round = latest_snapshot
      
    all_local_losses = []
    val_mae_losses = []
    val_mape_losses = []
    val_rmse_losses = []
    train_keys_logs = []
    
    curve_ratios_iter = iter(self.args.curve_ratios) if self.args.curve_ratios else None
    
    # TODO: use validation result to terminate global training?
    for global_round in range(starting_global_round, self.args.global_rounds):
      
      # Reload data if it is time to get new ratio of curves
      if curve_ratios_iter and global_round % self.args.curve_ratio_iter_rounds == 0:
        ratio = next(curve_ratios_iter, self.args.curve_ratios[-1])
        self.train_set = MITStanfordFeaturesDataset(self.args.data_dir,
                                                    data_keys=TRAIN_DATA_KEYS,
                                                    min_curve_len=self.args.min_curve_len,
                                                    future_interval=self.args.future_interval,
                                                    downsample_ratio=self.args.downsample_ratio,
                                                    curve_ratio=ratio)
        print(f'Loaded training data with ratio: {ratio}, Time passed: {timedelta(seconds=(time.time() - self.start_timestamp))}')  
      elif self.args.use_preprocess_curve_ratios and global_round % self.args.curve_ratio_iter_rounds == 0:
        preprocessed_curve_ratio_idx = global_round // self.args.curve_ratio_iter_rounds
        self.train_set = MITStanfordFeaturesDataset(self.args.data_dir,
                                                    data_keys=TRAIN_DATA_KEYS,
                                                    min_curve_len=self.args.min_curve_len,
                                                    future_interval=self.args.future_interval,
                                                    downsample_ratio=self.args.downsample_ratio,
                                                    preprocessed_curve_ratio_idx=preprocessed_curve_ratio_idx)
        print(f'Loaded training data with preprocessed curve ratio idx: {preprocessed_curve_ratio_idx}, Time passed: {timedelta(seconds=(time.time() - self.start_timestamp))}')  
      
      self.model.train()
      local_weights = []
      local_data_sizes = []
      local_losses = []
      client_selection_criteria_values = []

      if (self.args.client_select_method == 'random'):
        battery_keys = self.train_set.battery_idxs.keys()
        train_keys = random.sample(battery_keys, round(len(battery_keys)*self.args.client_fraction))
      else:        
        sorted_battery_keys = list(map(lambda item: item[0], sorted(self.train_set.battery_stats.items(), key=lambda item: item[1][self.args.client_select_method], reverse=self.args.client_select_sort_reverse)))
        train_keys = sorted_battery_keys[:round(len(sorted_battery_keys)*self.args.client_fraction)]
      
      for i, battery in enumerate(train_keys):
        data_idxs = self.train_set.battery_idxs[battery]
        print(f'Round {global_round+1}/{self.args.global_rounds} - [{i+1}/{len(train_keys)}] Start training on battery: {battery}')
        client_trainer = ClientTrainer(self.args, copy.deepcopy(self.model), self.device, self.train_set, data_idxs)
        weights, loss = client_trainer.train_one_round()
        local_weights.append(copy.deepcopy(weights))
        local_data_sizes.append(self.train_set.battery_data_sizes[battery])
        local_losses.append(copy.deepcopy(loss))
        if self.args.aggregation_method == 'cs_criteria_weighted_average':
          client_selection_criteria_values.append(self.train_set.battery_stats[battery][self.args.client_select_method])
        print(f'Round {global_round+1}/{self.args.global_rounds} - [{i+1}/{len(train_keys)}] Finished training on battery: {battery} | Loss: {loss:.4f} | Clients avg. loss: {sum(local_losses) / len(local_losses):.4f}')
      
      if self.args.aggregation_method == 'average':
        global_weights = self._average_weight(local_weights)
      elif self.args.aggregation_method == 'weighted_average':
        global_weights = self._weighted_average_weight(local_weights, local_data_sizes)
      elif self.args.aggregation_method == 'cs_criteria_weighted_average':
        global_weights = self._weighted_average_weight_by_cs_criteria(local_weights, client_selection_criteria_values)
      
      self.model.load_state_dict(global_weights)
      all_local_losses.append(local_losses)
      train_keys_logs.append(train_keys)
      
      val_mae_loss, val_mape_loss, val_rmse_loss = self._do_eval()      
      print(f'Round {global_round+1}/{self.args.global_rounds}: Time passed: {timedelta(seconds=(time.time() - self.start_timestamp))} Val MAE: {val_mae_loss:.4f} | Val MAPE: {val_mape_loss:.4f} | Val RMSE: {val_rmse_loss:.4f}')
      print('='*100)
      val_mae_losses.append(val_mae_loss)
      val_mape_losses.append(val_mape_loss)
      val_rmse_losses.append(val_rmse_loss)
      
      self.save_snapshot_model(global_round+1)
      
    os.makedirs(f'results/{self.args.exp_name}', exist_ok=True)
    json.dump(all_local_losses, open(f'results/{self.args.exp_name}/all_local_losses.json', 'w'))
    json.dump(val_mae_losses, open(f'results/{self.args.exp_name}/val_mae_losses.json', 'w'))
    json.dump(val_mape_losses, open(f'results/{self.args.exp_name}/val_mape_losses.json', 'w'))
    json.dump(val_rmse_losses, open(f'results/{self.args.exp_name}/val_rmse_losses.json', 'w'))
    json.dump(train_keys_logs, open(f'results/{self.args.exp_name}/train_keys_logs.json', 'w'))

  def _do_eval(self):
      self.model.eval()
      val_mae_loss = 0
      val_mape_loss = 0
      val_rmse_loss = 0
      for x, x_feat, y, y_feat, x_lens, y_lens, *_ in self.val_loader:
        x = x.float().to(self.device)
        x_feat = x_feat.float().to(self.device)
        y = y.float().to(self.device)
        y_feat = y_feat.float().to(self.device)
        
        pred = self.model(x, x_feat, y_feat, x_lens, y_lens)
        
        mae_loss = self.mae(pred, y)        
        val_mae_loss += mae_loss.item()
        mape_loss = self.mape(pred, y)
        val_mape_loss += mape_loss.item()
        rmse_loss = self.rmse(pred, y)
        val_rmse_loss += rmse_loss.item()
        
      val_mae_loss /= len(self.val_loader)
      val_mape_loss /= len(self.val_loader)
      val_rmse_loss /= len(self.val_loader)
      return val_mae_loss,val_mape_loss,val_rmse_loss
    
  def _weighted_average_weight_by_cs_criteria(self, weights, criteria_values):
    criteria_values = torch.tensor(criteria_values).to(self.device)
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
      weights_avg[key] = torch.zeros_like(weights_avg[key])
      for i in range(len(weights)):
        if self.args.client_select_sort_reverse:
          weights_avg[key] += torch.div(torch.mul(weights[i][key], criteria_values[i]), sum(criteria_values))
        else:
          weights_avg[key] += torch.div(torch.div(weights[i][key], criteria_values[i]), sum(torch.div(1, criteria_values)))
    return weights_avg
      
  def _weighted_average_weight(self, weights, data_sizes):
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
      weights_avg[key] = torch.zeros_like(weights_avg[key])
      for i in range(len(weights)):
        weights_avg[key] += torch.mul(torch.div(data_sizes[i], sum(data_sizes)), weights[i][key])
    return weights_avg
      
  def _average_weight(self, weights):
    weights_avg = copy.deepcopy(weights[0])
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    return weights_avg
  
  def save_snapshot_model(self, round):
    os.makedirs(f'results/{self.args.exp_name}', exist_ok=True)
    os.makedirs(f'results/{self.args.exp_name}/model_snapshot', exist_ok=True)
    torch.save(self.model.state_dict(), f'results/{self.args.exp_name}/model_snapshot/{round}.pth')
  
  def save_model(self):
    os.makedirs(f'results/{self.args.exp_name}', exist_ok=True)
    torch.save(self.model.state_dict(), f'results/{self.args.exp_name}/model.pth')
      
  def load_model(self):
    self.model.load_state_dict(torch.load(f'results/{self.args.exp_name}/model.pth'))
    
  def test_late_start(self):
    self.model.eval()
    test_time = datetime.now()
    self._do_dataset_test('B1_0.1_0.5', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, 'test_late_start', 0.5, 0.1, test_time=test_time)
    self._do_dataset_test('B1_0.2_0.6', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, 'test_late_start', 0.6, 0.2, test_time=test_time)
    self._do_dataset_test('B1_0.3_0.7', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, 'test_late_start', 0.7, 0.3, test_time=test_time)
    
    self._do_dataset_test('B2_0.1_0.5', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, 'test_late_start', 0.5, 0.1, test_time=test_time)
    self._do_dataset_test('B2_0.2_0.6', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, 'test_late_start', 0.6, 0.2, test_time=test_time)
    self._do_dataset_test('B2_0.3_0.7', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, 'test_late_start', 0.7, 0.3, test_time=test_time)
    
    self._do_dataset_test('B3_0.1_0.5', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, 'test_late_start', 0.5, 0.1, test_time=test_time)
    self._do_dataset_test('B3_0.2_0.6', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, 'test_late_start', 0.6, 0.2, test_time=test_time)
    self._do_dataset_test('B3_0.3_0.7', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, 'test_late_start', 0.7, 0.3, test_time=test_time)
    
    
  def test(self):
    self.model.eval()
    test_time = datetime.now()
    self._do_dataset_test('All', VAL_DATA_KEYS if self.args.use_val_set_for_testing else TEST_DATA_KEYS, test_time=test_time)
    
    self._do_dataset_test('B1', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, test_time=test_time)
    self._do_dataset_test('B2', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, test_time=test_time)
    self._do_dataset_test('B3', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, test_time=test_time)
    
    self._do_dataset_test('Early_input_B1', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, input_ratio=0.3, test_time=test_time)
    self._do_dataset_test('Middle_input_B1', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, input_ratio=0.5, test_time=test_time)
    self._do_dataset_test('Late_input_B1', VAL_DATA_B1_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B1_KEYS, input_ratio=0.7, test_time=test_time)
    
    self._do_dataset_test('Early_input_B2', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, input_ratio=0.3, test_time=test_time)
    self._do_dataset_test('Middle_input_B2', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, input_ratio=0.5, test_time=test_time)
    self._do_dataset_test('Late_input_B2', VAL_DATA_B2_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B2_KEYS, input_ratio=0.7, test_time=test_time)
    
    self._do_dataset_test('Early_input_B3', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, input_ratio=0.3, test_time=test_time)
    self._do_dataset_test('Middle_input_B3', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, input_ratio=0.5, test_time=test_time)
    self._do_dataset_test('Late_input_B3', VAL_DATA_B3_KEYS if self.args.use_val_set_for_testing else TEST_DATA_B3_KEYS, input_ratio=0.7, test_time=test_time)
    
  def _do_dataset_test(self, name, data_keys, test_category='test', input_ratio=None, start_ratio=0, save=True, test_time=datetime.now()):
    test_set = MITStanfordFeaturesDataset(self.args.data_dir,
                                          data_keys=data_keys,
                                          min_curve_len=self.args.min_curve_len,
                                          future_interval=self.args.future_interval,
                                          downsample_ratio=self.args.downsample_ratio,
                                          input_ratio=input_ratio,
                                          start_ratio=start_ratio)
    test_loader = DataLoader(test_set, batch_size=self.args.val_batch_size, shuffle=False)
    test_mae_loss = 0
    test_mape_loss = 0
    test_rmse_loss = 0
    x_list,y_list,pred_list = [],[],[]
    for x, x_feat, y, y_feat, x_lens, y_lens, *_ in test_loader:
      x = x.float().to(self.device)
      x_feat = x_feat.float().to(self.device)
      y = y.float().to(self.device)
      y_feat = y_feat.float().to(self.device)
      pred = self.model(x, x_feat, y_feat, x_lens, y_lens)
      mae_loss = self.mae(pred, y)
      test_mae_loss += mae_loss.item()     
      mape_loss = self.mape(pred, y)
      test_mape_loss += mape_loss.item()
      rmse_loss = self.rmse(pred, y)
      test_rmse_loss += rmse_loss.item()
      x_list.append(x)
      y_list.append(y)
      pred_list.append(pred)
      
    test_mae_loss /= len(test_loader)
    test_mape_loss /= len(test_loader)
    test_rmse_loss /= len(test_loader)
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    pred = torch.cat(pred_list, dim=0)
    print(f'[{name}] test MAE: {test_mae_loss:.4f}, test MAPE: {test_mape_loss:.4f}, test RMSE: {test_rmse_loss:.4f}')
    
    if save:
      test_dir = f'results/{self.args.exp_name}/{test_category}_{test_time.strftime("%Y%m%d%H%M%S")}'
      os.makedirs(test_dir, exist_ok=True)
      json.dump(pred.cpu().detach().numpy().tolist(), open(f'{test_dir}/{name}_pred.json', 'w'))
      json.dump(y.cpu().detach().numpy().tolist(), open(f'{test_dir}/{name}_true.json', 'w'))
      json.dump(x.cpu().detach().numpy().tolist(), open(f'{test_dir}/{name}_x.json', 'w'))