import os.path as osp
import xml.etree.ElementTree as ET
from mmdet.core import eval_map, eval_recalls
import mmcv
import numpy as np
from .registry import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module
class InsectDataset(XMLDataset):
    """
    Reader for the Insects dataset in VOC format.
    """
    CLASSES = ('Boerner', 'Leconte', 'Linnaeus',
                'acuminatus', 'armandi', 'coleoptera', 'linnaeus')

    def __init__(self, **kwargs):
        super(InsectDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'images/{}.jpeg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'annotations/xmls',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height
                )
            )
        return img_infos
    
    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'annotations/xmls',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0, ), dtype=np.int64)
        else:
            bboxes = np.array(bboxes, dtype=np.float32) - 1
            labels = np.array(labels, dtype=np.int64)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32) - 1
        ann = dict(
            bboxes=bboxes,
            labels=labels,
            bboxes_ignore=bboxes_ignore)
        return ann
        
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            ds_name = self.CLASSES
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results['recall@{}@{}'.format(num, iou)] = recalls[i,
                                                                            j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
        return eval_results