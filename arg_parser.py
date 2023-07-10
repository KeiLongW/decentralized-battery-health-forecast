import argparse
from datetime import datetime


def parse_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--exp_name", type=str, help="Experiment name", default=datetime.now().strftime('%Y%m%d_%H%M%S'))
  arg_parser.add_argument("--client_batch_size", type=int, help="Batch size in client training", default=256)
  arg_parser.add_argument("--client_epochs", type=int, help="Epoch of local client training", default=30)
  arg_parser.add_argument("--client_learning_rate", type=float, help="Learning rate in client training", default=0.001)
  arg_parser.add_argument("--val_batch_size", type=int, help="Batch size in server validation", default=256)
  arg_parser.add_argument("--data_dir", type=str, help="Directory of data", default="data/MIT Stanford battery dataset/")
  arg_parser.add_argument("--min_curve_len", type=int, help="Minimum length of input degradation curve", default=100)
  arg_parser.add_argument("--downsample_ratio", type=float, help="Downsample ratio of input degradation curve", default=1)
  arg_parser.add_argument("--future_interval", type=int, help="Sampling interval of future degradation curve (output)", default=1)
  arg_parser.add_argument("--global_rounds", type=int, help="Number of global rounds", default=50)
  arg_parser.add_argument("--step_lr_step_size", type=int, help="Step learning rate scheduler step size", default=1)
  arg_parser.add_argument("--step_lr_gamma", type=float, help="Step learning rate scheduler gamma", default=1)
  arg_parser.add_argument("--client_optimizer", type=str, help="Client optimizer", default="adam")
  arg_parser.add_argument("--client_sgd_optimizer_momentum", type=float, help="Client SGD optimizer momentum", default=0.9)
  
  arg_parser.add_argument("--lstm_layers", type=int, help="Number of LSTM layers", default=1)
  arg_parser.add_argument("--lstm_bidirectional", action="store_true", help="Enable bidirectional LSTM")
  
  arg_parser.add_argument("--train_baseline", action="store_true", help="Enable training baseline model")
  arg_parser.add_argument("--baseline_epoch", type=int, help="Number of epochs for baseline training", default=50)
  arg_parser.add_argument("--baseline_batch_size", type=int, help="Batch size for baseline training", default=256)
  arg_parser.add_argument("--baseline_learning_rate", type=float, help="Learning rate for baseline training", default=0.001)
  
  arg_parser.add_argument("--encoder_hidden_dim", type=int, help="Encoder hidden layer dimension", default=64)
  arg_parser.add_argument("--decoder_hidden_dim", type=int, help="Decoder hidden layer dimension", default=64)
  
  arg_parser.add_argument("--curve_ratios", nargs="+", type=float, help="Curve ratios of each curve extracted from the dataset", default=None)
  arg_parser.add_argument("--curve_ratio_iter_rounds", type=int, help="Number of global rounds for each curve ratio", default=10)
  arg_parser.add_argument("--use_preprocess_curve_ratios", action="store_true", help="Use curve ratios settings from preprocessed data")
  
  arg_parser.add_argument("--client_fraction", type=float, help="Fraction of clients selected in each global round", default=0.5)
  arg_parser.add_argument("--client_select_method", type=str, help="Client selection method", default="random")
  arg_parser.add_argument("--client_select_sort_reverse", action="store_true", help="Reverse the sorting order of client selection")
  
  arg_parser.add_argument("--fed_algorithm", type=str, help="Federated learning algorithm", default="FedAvg")
  arg_parser.add_argument("--fedprox_mu", type=float, help="Federated learning algorithm Fed Prox proximal term constant", default=0.01)
  arg_parser.add_argument("--aggregation_method", type=str, help="Aggregation method", default="average")
  
  arg_parser.add_argument("--testing", action="store_true", help="Enable testing mode")
  arg_parser.add_argument("--use_val_set_for_testing", action="store_true", help="Use validation set for testing")
  arg_parser.add_argument("--testing_late_start", action="store_true", help="Enable late start testing mode")
  
  arg_parser.add_argument("--continue_training", action="store_true", help="Enable continue training mode")
  
  return arg_parser.parse_args()