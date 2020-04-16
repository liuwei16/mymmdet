import os
import mmcv
import time
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import get_root_logger
import _pickle as cPickle
import ipdb

config = 'config_example.py'
work_dir = 'work_dirs/retinanet_insects'
res_dir = os.path.join(work_dir, 'resdir')
mmcv.mkdir_or_exist(res_dir)
cfg = mmcv.Config.fromfile(config)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = os.path.join(res_dir, '{}.log'.format(timestamp))
logger = get_root_logger(log_file, 'INFO')
# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset=dataset,
    imgs_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False)

for i in range(1, 8):
    respath = os.path.join(res_dir, '{}.pkl'.format(i))
    if os.path.isfile(respath):
        with open(respath, 'rb') as fid:
            outputs = cPickle.load(fid)
    else:
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        checkpoint_path = os.path.join(work_dir, 'epoch_{}.pth'.format(i))
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
        mmcv.dump(outputs, respath)
    eval_results = dataset.evaluate(outputs, 'mAP')
    logger.info('{}:{}'.format(i, eval_results['mAP']))