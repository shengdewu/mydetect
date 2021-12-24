# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    # "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    # "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    # "coco_2014_valminusminival": (
    #     "coco/val2014",
    #     "coco/annotations/instances_valminusminival2014.json",
    # ),
    # "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    # "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    # "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    # "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    # "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS_XINTU = {}
_PREDEFINED_SPLITS_XINTU["xintu"] = {
    "coco_2014_train": ("human.segmentation.coco.data/train", "human.segmentation.coco.data/annotations/instances_train.json"),
    "coco_2014_val": ("human.segmentation.coco.data/val", "human.segmentation.coco.data/annotations/instances_val.json"),
}

_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    # for (
    #     prefix,
    #     (panoptic_root, panoptic_json, semantic_root),
    # ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
    #     prefix_instances = prefix[: -len("_panoptic")]
    #     instances_meta = MetadataCatalog.get(prefix_instances)
    #     image_root, instances_json = instances_meta.image_root, instances_meta.json_file
    #     # The "separated" version of COCO panoptic segmentation dataset,
    #     # e.g. used by Panoptic FPN
    #     register_coco_panoptic_separated(
    #         prefix,
    #         _get_builtin_metadata("coco_panoptic_separated"),
    #         image_root,
    #         os.path.join(root, panoptic_root),
    #         os.path.join(root, panoptic_json),
    #         os.path.join(root, semantic_root),
    #         instances_json,
    #     )
    #     # The "standard" version of COCO panoptic segmentation dataset,
    #     # e.g. used by Panoptic-DeepLab
    #     register_coco_panoptic(
    #         prefix,
    #         _get_builtin_metadata("coco_panoptic_standard"),
    #         image_root,
    #         os.path.join(root, panoptic_root),
    #         os.path.join(root, panoptic_json),
    #         instances_json,
    #     )
    #

def register_all_xintu(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_XINTU.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


def draw(root, img_path, json_path, bbox_path, mask_path):
    import json
    import cv2
    import numpy as np
    import pycocotools._mask as _mask

    os.makedirs(os.path.join(root, bbox_path), exist_ok=True)
    os.makedirs(os.path.join(root, mask_path), exist_ok=True)

    with open('{}/annotations/{}'.format(root, json_path), 'r') as f:
        dataset_dict = json.load(f)

    images = dataset_dict['images']
    annotations = dataset_dict['annotations']

    id_images = dict()
    for image in images:
        id_images[image['id']] = image

    id_anns = dict()
    for ann in annotations:
        if id_anns.get(ann['image_id'], None) is None:
            id_anns[ann['image_id']] = list()
        id_anns[ann['image_id']].append(ann)

    for id, image in id_images.items():
        print(id)
        file_name = image['file_name']
        image_path = os.path.join(root, img_path, file_name)
        img = cv2.imread(image_path)
        mask = img.copy()
        tmp = img.copy()
        anns = id_anns.get(id, [])
        ann_ids = 0
        for ann in anns:
            ann_ids += 1
            c = (np.random.random((1, 3))*255).astype(np.int).tolist()[0]
            polys = list()

            if isinstance(ann['segmentation'], dict):
                seg = _mask.decode(ann['segmentation'])
                print('rel')
                continue

            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2)).astype(np.int)
                polys.append(poly)
            cv2.fillPoly(mask, polys, c)
            cv2.imwrite(os.path.join(root, mask_path, '{}-ann{}-seg{}.{}'.format(file_name[:file_name.find('.jpg')], ann_ids, len(polys), file_name[file_name.find('jpg'):])), mask)
            if len(polys) > 1:
                p = 0
                for poly in polys:
                    p += 1
                    cv2.fillPoly(tmp, [poly], c)
                    cv2.imwrite(os.path.join(root, mask_path, '{}-ann{}-seg{}-p{}.{}'.format(file_name[:file_name.find('.jpg')], ann_ids, len(polys), p, file_name[file_name.find('jpg'):])), tmp)
            bbox = ann['bbox'] # x, y w, h
            cv2.putText(img, 'person', (int(bbox[0]), int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), c, 1, 4)

        cv2.imwrite(os.path.join(root, bbox_path, file_name), img)


def xintu2coco(root):
    import json
    import cv2
    import numpy as np
    import shutil

    with open('{}/annotations/photo_cut_point_coco.json'.format(root), 'r') as f:
        dataset_dict = json.loads(f.read().replace('][', ','))

    images = dataset_dict['images']
    annotations = dataset_dict['annotations']
    bbox_id_start = 0
    image_id_start = 0
    modify_dataset = dict()

    modify_dataset['info'] = dataset_dict['info']
    modify_dataset['licenses'] = dataset_dict['licenses']
    modify_dataset['images'] = list()
    modify_dataset['annotations'] = list()

    images_ids = dict()
    annotations_ids = dict()
    anns_ids = list()
    for image, annotation in zip(images, annotations):
        cimage = image.copy()
        image_id_start += 1
        cimage['id'] = image_id_start
        file_name = image['file_name']
        image_path = os.path.join(root, 'images', file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('cannot identify image file {}'.format(image_path))
            continue

        image_h, image_w = img.shape[0:2]
        cimage["width"] = image_w
        cimage["height"] = image_h

        modify_dataset['images'].append(cimage)

        if images_ids.get(cimage['id'], None) is not None:
            print('{}-{} duplication'.format(image_path, cimage['id']))
        images_ids[cimage['id']] = cimage.copy()

        try:
            for ants in annotation:
                cant = ants.copy()
                bbox_id_start += 1
                cant['image_id'] = cimage['id']
                cant['iscrowd'] = 0
                cant['id'] = bbox_id_start
                modify_dataset['annotations'].append(cant)

                if annotations_ids.get(cimage['id'], None) is None:
                    annotations_ids[cimage['id']] = list()

                annotations_ids[cimage['id']].append(cant.copy())

                anns_ids.append(cant['id'])

        except Exception as err:
            for k, v in annotation.items():
                cant = v.copy()
                bbox_id_start += 1
                cant['image_id'] = cimage['id']
                cant['iscrowd'] = 0
                cant['id'] = bbox_id_start
                modify_dataset['annotations'].append(cant)

                if annotations_ids.get(cimage['id'], None) is None:
                    annotations_ids[cimage['id']] = list()

                annotations_ids[cimage['id']].append(cant)

                anns_ids.append(cant['id'])

    modify_dataset['categories'] = list()
    category = dict()
    category['supercategory'] = 'person'
    category['id'] = 1
    category['name'] = 'person'
    modify_dataset['categories'].append(category)

    with open('{}/annotations/instances_total.json'.format(root), 'w') as f:
        json.dump(modify_dataset, f)

    train_dataset = dict()
    train_dataset['info'] = modify_dataset['info']
    train_dataset['licenses'] = modify_dataset['licenses']
    train_dataset['images'] = list()
    train_dataset['annotations'] = list()
    train_dataset['categories'] = modify_dataset['categories']

    val_dataset = dict()
    val_dataset['info'] = modify_dataset['info']
    val_dataset['licenses'] = modify_dataset['licenses']
    val_dataset['images'] = list()
    val_dataset['annotations'] = list()
    val_dataset['categories'] = modify_dataset['categories']

    os.makedirs(os.path.join(root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root, 'val'), exist_ok=True)

    ids = list(images_ids.keys())
    val_ids = np.random.choice(ids, int(len(ids)*0.1))
    for id in set(ids):
        img_info = images_ids[id]
        file_name = img_info['file_name']
        img_path = os.path.join(root, 'images', file_name)
        if id in val_ids:
            val_dataset['images'].append(img_info)
            val_dataset['annotations'].extend(annotations_ids[id])
            shutil.copy2(img_path, os.path.join(root, 'val', file_name))
        else:
            train_dataset['images'].append(img_info)
            train_dataset['annotations'].extend(annotations_ids[id])
            shutil.copy2(img_path, os.path.join(root, 'train', file_name))

    with open('{}/annotations/instances_val.json'.format(root), 'w') as f:
        json.dump(val_dataset, f)

    with open('{}/annotations/instances_train.json'.format(root), 'w') as f:
        json.dump(train_dataset, f)

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS_ROOT", "/mnt/data/coco.data")
    data_type = os.getenv("DETECTRON2_DATASETS_TYPE", "coco")
    print('register data {}-{}'.format(_root, data_type))
    if data_type == 'xintu':
        register_all_xintu(_root)
    else:
        register_all_coco(_root)
    # xintu2coco('/mnt/data/data.set/xintu.data/human.segmentation.coco.data')
    # draw('/mnt/data/data.set/xintu.data/human.segmentation.coco.data', 'train', 'instances_train.json', 'check/bbox', 'check/mask')
    # draw('/mnt/data/data.set/xintu.data/human.segmentation.coco.data', 'val', 'instances_val.json', 'check/bbox', 'check/mask')
    # register_all_lvis(_root)
    # register_all_cityscapes(_root)
    # register_all_cityscapes_panoptic(_root)
    # register_all_pascal_voc(_root)
    # register_all_ade20k(_root)
