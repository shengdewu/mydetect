#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import torch
import torchvision
pwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pwd)
sys.path.append(os.path.dirname(__file__))
from fvcore.common.checkpoint import Checkpointer
from detectron2.data import DatasetMapper
from detectron2.engine import default_argument_parser
from projects.PointRend.train_net import (Trainer, setup)
import cv2


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


import json
import time
import numpy as np
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

if __name__ == "__main__":
    with open('/home/shengdewu/Documents/colors.map.json') as hc:
        color_map = list(json.load(hc).values())

    coco_meta = _get_coco_instances_meta()
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    model, data_transform = create_model(args)

    root_path = '/mnt/data/xintu.data/human.segmetn.coco.data/val'
    #root_path ='/mnt/data/coco.data/coco/test2014'

    total_time = 0
    total_index = 0
    for img_name in os.listdir(root_path):
        stat_time = time.time()
        try:
            instances = inference(model, data_transform, os.path.join(root_path, img_name))
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

        img = cv2.imread(os.path.join(root_path, img_name))

        raw = img.copy()
        total = scores.shape[0]
        print('total detect:{}'.format(total))
        for i in range(total):
            pred_box = pred_boxes[i].tensor[0].numpy()
            score = float(scores[i].numpy())
            if score < 0.7:
                continue
            pred_class = pred_classes[i].numpy()
            pred_mask = pred_masks[i].numpy()
            color = color_map[i+10][0]
            img[pred_mask, 0] = color[0]
            img[pred_mask, 1] = color[1]
            img[pred_mask, 2] = color[2]
            cv2.putText(img, '{}:{}'.format(coco_meta['thing_classes'][pred_class], round(score, 2)), (int(pred_box[0]), int(pred_box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[200][0], 1)
            cv2.rectangle(img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), color, 3, 1)
            #cv2.imwrite('/mnt/data/human.model/pointrend.xintu/inference/{}-m{}{}'.format(img_name[:img_name.rfind('.')], i, img_name[img_name.rfind('.'):]), pred_mask.astype(img.dtype)*255)

            # mask = cv2.bitwise_and(img, img, mask=pred_mask.astype(img.dtype))
            # #cv2.floodFill(img, mask, (0, 0), color_map[i+10][0], color_map[i+10][0], color_map[i+10][0], cv2.FLOODFILL_FIXED_RANGE)
        cv2.imwrite('/mnt/data/human.model/pointrend.xintu/inference/{}'.format(img_name), np.hstack((raw, img)))

    print('cost {} in {} images: avg: {}'.format(total_time, total_index, total_time/total_index))



