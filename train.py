from __future__ import division
import argparse
import copy
import os
import os.path as osp
import glob
import time
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
import ipdb
# path = 'data/insects/val/images/*.jpeg'
# with open('data/insects/val.txt', 'w') as f:
#     for i in glob.glob(path):
#         f.write(osp.splitext(osp.basename(i))[0]+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description='mydet')
    parser.add_argument('--config', default='config_example.py', help='train config file path')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    ipdb.set_trace(context=10)
    for i in range(len(dataset)):
        data = dataset[i]




if __name__ == '__main__':
    main()