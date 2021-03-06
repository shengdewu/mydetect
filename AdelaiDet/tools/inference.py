#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import torch
pwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pwd)
sys.path.append(os.path.dirname(__file__))
from fvcore.common.checkpoint import Checkpointer
import cv2

import json
import time
import numpy as np
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from tools.train_net import Trainer
from adet.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data.dataset_mapper import DatasetMapper
import pycocotools.mask
from detectron2.data.datasets.coco import load_coco_json


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def create_model(args):
    cfg = setup(args)
    model = Trainer.build_model(cfg)
    checkpointer = Checkpointer(model)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    data_transform = DatasetMapper(cfg, False)
    model.training = False
    model.eval()
    return model.to(cfg.MODEL.DEVICE), data_transform


@torch.no_grad()
def inference(model, data_transform, img_path):
    img_dict = dict()
    img_dict['file_name'] = img_path
    img_dict['image_id'] = 1
    timg = data_transform(img_dict)
    outputs = model([timg])
    instances = outputs[0]['instances']
    return instances

def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


if __name__ == "__main__":

    coco_meta = _get_coco_instances_meta()
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    model, data_transform = create_model(args)

    #root_path = '/mnt/data/xintu.data/human.segmetn.coco.data'
    #'/mnt/data/xintu.data/human.segmetn.coco.data', ['instances_val.json'], ['annotations', 'val']
    root_path ='/mnt/data/data.set/coco.data/coco'
    dataset_dicts = load_coco_json(os.path.join(root_path, 'annotations', 'instances_val2014.json'), os.path.join(root_path, 'val2014'), 'coco')

    total_time = 0
    total_index = 0
    for dataset in dataset_dicts:
        file_name = dataset['file_name']
        print(file_name)
        stat_time = time.time()
        try:
            instances = inference(model, data_transform, file_name)
        except Exception as err:
            print(err)
            continue
        total_time += time.time() - stat_time
        total_index += 1
        pred_boxes = instances.get('pred_boxes')  # n *4
        scores = instances.get('scores')  # n
        pred_classes = instances.get('pred_classes')  # n
        pred_masks = instances.get('pred_masks')  # n * h * w
        h, w = instances.image_size

        img = cv2.imread(file_name)

        raw = img.copy()
        total = scores.shape[0]
        print('total detect:{}'.format(total))
        for i in range(total):
            pred_box = pred_boxes[i].tensor[0].numpy()
            score = float(scores[i].numpy())
            if score < 0.7:
                continue
            pred_class = pred_classes[i].numpy()

            pred_mask = np.asarray(pred_masks[i], dtype=np.uint8)

            polygons, has_holes = mask_to_polygons(pred_mask)

            color = coco_meta['thing_colors'][pred_class]
            thing = coco_meta['thing_classes'][pred_class]

            polygons = [np.asarray(p).astype(np.int).reshape(-1, 2) for p in polygons]

            cv2.polylines(img, polygons, isClosed=False, color=color, thickness=3)
            # cv2.fillPoly(img, [poly_box], lineType=cv2.LINE_8, color=color)
            cv2.putText(img, '{}/{}'.format(thing, round(score, 2)), (int(polygons[0][0][0]), int(polygons[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

            # mask = cv2.bitwise_and(img, img, mask=pred_mask.astype(img.dtype))
            # #cv2.floodFill(img, mask, (0, 0), color_map[i+10][0], color_map[i+10][0], color_map[i+10][0], cv2.FLOODFILL_FIXED_RANGE)
        for ann in dataset['annotations']:
            color = coco_meta['thing_colors'][ann['category_id']]
            thing = coco_meta['thing_classes'][ann['category_id']]

            segm = ann['segmentation']

            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).astype(np.int).reshape(-1, 2) for p in segm]
            elif isinstance(segm, dict):
                # RLE
                mask = pycocotools.mask.decode(segm)
                polygons, has_holes = mask_to_polygons(mask)
                polygons = [np.asarray(p).astype(np.int).reshape(-1, 2) for p in polygons]
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )

            cv2.polylines(raw, polygons, isClosed=False, color=color, thickness=3)
            cv2.putText(raw, '{}'.format(thing), (int(polygons[0][0][0]), int(polygons[0][0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

        cv2.imwrite('/mnt/data/train.output/solov2.output/{}'.format(file_name[file_name.rfind('/')+1:]), np.hstack((raw, img)))

    print('cost {} in {} images: avg: {}'.format(total_time, total_index, total_time/total_index))



