import sys
import os
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
# from scoring_program.summary import full_evaluation
from src.evaluation.summary import full_evaluation
from src.evaluation.strategies import log_return_to_price

import ml_collections
import yaml

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# input_dir = "../input"
# output_dir = "../output"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the config file
config_dir = os.path.join(script_dir, 'config.yaml')

with open(config_dir) as file:
    config = ml_collections.ConfigDict(yaml.safe_load(file))

torch.manual_seed(config.seed)

# input_dir = "../input"
# output_dir = "../output"
submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a file to store scores
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    # Load test data
    truth_file = os.path.join(truth_dir, "truth_log_return.pkl")
    if not os.path.isfile(truth_file):
        raise Exception('Data not supplied')
    with open(truth_file, "rb") as f:
        truth_log_return = torch.tensor(pickle.load(f)).float().to(config.device)

    truth_file = os.path.join(truth_dir, "truth_price.pkl")
    if not os.path.isfile(truth_file):
        raise Exception('Data not supplied')
    with open(truth_file, "rb") as f:
        truth_init_price = torch.tensor(pickle.load(f)).float().to(config.device)

    # Load fake data
    fake_file = os.path.join(submit_dir, "fake_log_return.pkl")
    if not os.path.isfile(fake_file):
        raise Exception('Data not supplied, make sure the zip file contains the file named fake_log_return.pkl')
    with open(fake_file, "rb") as f:
        fake_log_return = torch.tensor(pickle.load(f)).float().to(config.device)

    truth_log_return = truth_log_return[:1800]
    truth_init_price = truth_init_price[:1800]
    fake_log_return = fake_log_return[:1800]

    truth_price = log_return_to_price(truth_log_return, truth_init_price)
    fake_price = log_return_to_price(fake_log_return, truth_init_price)

    res_dict = {"var_mean" : 0., "es_mean": 0., "max_drawback_mean": 0., "cumulative_pnl_mean": 0.}

    # Do final evaluation
    num_strat = 4
    for strat_name in ['equal_weight', 'mean_reversion', 'trend_following', 'vol_trading']:
        subres_dict = full_evaluation(fake_price, truth_price, config, strat_name = strat_name)
        for k in res_dict:
            res_dict[k] += subres_dict[k] / num_strat
    print(res_dict)

    output_file.write(
        "cumulative_pnl_mean:%0.5f" % (res_dict["cumulative_pnl_mean"]))
    output_file.write("\n")
    output_file.write(
        "max_drawback_mean:%0.5f" % (res_dict["max_drawback_mean"]))
    output_file.write("\n")
    output_file.write("var_mean:%0.5f" % (res_dict["var_mean"]))
    output_file.write("\n")
    output_file.write("es_mean:%0.5f" % (res_dict["es_mean"]))
    output_file.write("\n")

    output_file.close()