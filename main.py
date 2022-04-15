#!/usr/bin/env python
#-*- encoding: UTF-8 -*-
"""main.py: the only entrance of program"""
# python library: a-z
import argparse
import collections
import numpy as np
import torch


# user library
from utils.parse_config import ConfigParser
import trainer.data_loaders as module_data
from train import train
from test import test


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)




def main(config):
    """
    the whole stages of deep learning model:
    - prepare the data
    - train the model
    - test the model
    """
    # initialize logger
    logger = config.get_logger('main')

    # build model architecture, then print to console
    model = train(config)
    # test()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="conf/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("-s", default="start", type=str, 
                      help="running mode of program (default: start. option: resume, test)")
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

    
