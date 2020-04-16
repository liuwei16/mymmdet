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
from mmdet.datasets import build_dataset,build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import ipdb

def parse_args():
    parser = argparse.ArgumentParser(description='mydet')
    parser.add_argument('--config', default='config_example.py', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher =='none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    #create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    #init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    # logger = get_root_logger(log_file, cfg.log_level)

    dataset = build_dataset(cfg.data.train)

    dataloader = build_dataloader(dataset=dataset,
                     imgs_per_gpu=cfg.data.imgs_per_gpu,
                     workers_per_gpu=cfg.data.workers_per_gpu,
                     num_gpus=cfg.gpus,
                     dist=False)
    # ipdb.set_trace(context=10)
    for i,data in enumerate(dataloader):
        print(i)



if __name__ == '__main__':
    main()