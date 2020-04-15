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
import _pickle as cPickle
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
    
    widths, heigths = [], []
    areas = []
    img_ars = []
    box_ws, box_hs, box_ars, labels = [], [], [], []
    # ipdb.set_trace(context=10)
    for i in range(len(dataset)):
        if i%500==0:
            print(i)
        data = dataset[i]
        heigths.append(data['img'].shape[0])
        widths.append(data['img'].shape[1])
        img_ars.append(data['img'].shape[1]/data['img'].shape[0])
        box_w = data['gt_bboxes'][:,2]-data['gt_bboxes'][:,0]
        box_h = data['gt_bboxes'][:,3]-data['gt_bboxes'][:,1]
        box_ar = box_w/box_h
        box_ws.append(box_w)
        box_hs.append(box_h)
        box_ars.append(box_ar)
        labels.append(data['gt_labels'])
    # ipdb.set_trace(context=10)
    dict = {}
    dict['heigths'] = heigths
    dict['widths'] = widths
    dict['img_ars'] = img_ars
    dict['box_ws'] = np.concatenate(box_ws)
    dict['box_hs'] = np.concatenate(box_hs)
    dict['box_ars'] = np.concatenate(box_ars)
    dict['labels'] = np.concatenate(labels)
    with open('train_info.pkl', 'wb') as f:
        cPickle.dump(dict, f)
    



if __name__ == '__main__':
    main()