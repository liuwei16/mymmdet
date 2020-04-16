
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
sys.path.insert(0, '/home/aistudio/work/myenv/lib/python3.7/site-packages/')
# sys.path.insert(0, '/home/aistudio/work/mymmdet/')
# import _pickle as cPickle
# with open('work/mymmdet/train_info.pkl', 'rb') as f:
#     dict = cPickle.load(f)
# data = dict['labels']
# print(len(data))
plt.figure(figsize=(8, 8), dpi=80)
# plt.hist(data, bins=7)
# plt.grid(True, linestyle='--', alpha=0.8)
# plt.show()
# set(data)
from mmcv import Config
from mmdet.datasets import build_dataset
cfg = Config.fromfile('config_example.py')
dataset = build_dataset(cfg.data.train)
# for i in range(len(dataset)):
# colors = {i:np.random.randint(0,255,3)/255 for i in range(1,8)}
colors = ['r', 'g', 'b', 'k', 'y', 'pink', 'purple']
for i in range(10):
    data = dataset[0]
    img = data['img'].astype(np.int32)
    print(img.shape[:2])
# bboxes = data['gt_bboxes']
# labels = data['gt_labels']
# print(img.shape)
# plt.imshow(img)
# for i in range(len(bboxes)):
#     plt.gca().add_patch(plt.Rectangle((bboxes[i][0], bboxes[i][1]), bboxes[i][2]-bboxes[i][0], bboxes[i][3]-bboxes[i][1], 
#     fill=False, edgecolor=colors[labels[i]-1], linewidth=2))
# plt.show()
