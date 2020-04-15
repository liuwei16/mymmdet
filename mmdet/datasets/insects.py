import os.path as osp
import xml.etree.ElementTree as ET

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