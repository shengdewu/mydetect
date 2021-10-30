import os
import cv2
from torchvision.datasets import CocoDetection
from copy_paste import copy_paste_class
from copy_paste import CopyPaste
import albumentations as A
from typing import Dict, Callable

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        #self.coco.showAnns(target, draw_bbox=True)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            xmin, ymin, bw, bh = obj['bbox']
            xmin = np.clip(xmin, 0, w)
            ymin = np.clip(ymin, 0, w)
            if xmin + bw > w:
                bw = bw - (xmin + bw - w)
            if ymin + bh > h:
                bh = bh - (ymin + bh - h)
            bboxes.append([xmin, ymin, bw, bh] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes,
            'img_name':path
        }

        return output

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


def draw(image, masks, bboxes, convert=False):
    for mask in masks:
        polygons, has_holes = mask_to_polygons(mask)

        for polygon in polygons:
            poly_box = np.array([int(p + 0.5) for p in polygon], dtype=np.int).reshape(-1, 2)
            cv2.polylines(image, [poly_box], isClosed=False, color=(255, 255, 0), thickness=3)
            # cv2.fillPoly(img, [poly_box], lineType=cv2.LINE_8, color=color)
    for box in bboxes:
        if not convert:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 255), thickness=2, lineType=1)
        else:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), thickness=2, lineType=1)


def draw_anno(image, annos):
    for anno in annos:
        polygons = anno['segmentation']

        for polygon in polygons:
            poly_box = np.array([int(p + 0.5) for p in polygon], dtype=np.int).reshape(-1, 2)
            cv2.polylines(image, [poly_box], isClosed=False, color=(255, 255, 0), thickness=3)

        box = anno['bbox']
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 255), thickness=2, lineType=1)


import numpy as np
import time
import json
import tqdm
if __name__ == '__main__':

    class RandomCrop(A.RandomCrop):
        def __init__(self, height, width, always_apply=False, p=1.0, no_random=True):
            super(RandomCrop, self).__init__(height, width, always_apply, p=p)
            self.no_random = no_random
        @property
        def targets(self) -> Dict[str, Callable]:
            return {
                "image": self.apply,
                "mask": self.apply_to_mask,
                "masks": self.apply_to_masks,
                "bboxes": self.apply_to_bboxes,
                "keypoints": self.apply_to_keypoints,
                "paste_image": self.apply,
                "paste_masks": self.apply_to_masks,
                "paste_bboxes": self.apply_to_bboxes,
            }

        def get_params(self):
            if self.no_random:
                return {"h_start": 0, "w_start": 0}
            else:
                return {"h_start": np.random.random(), "w_start": np.random.random()}

    # img_root = '/mnt/data/coco.data/coco/val2014'
    # annFile = '/mnt/data/coco.data/coco/annotations/instances_val2014.json'

    annFile = '/mnt/data/xintu.data/human.segmetn.coco.data/annotations/instances_train.json'
    img_root = '/mnt/data/xintu.data/human.segmetn.coco.data/train'

    coccp = CocoDetectionCP(img_root,
                            annFile,
                            None)

    output_root = '/mnt/data/xintu.data/human.segmetn.copypaste.data'
    annotations = os.path.join(output_root, 'annotations')
    train = os.path.join(output_root, 'train')
    val = os.path.join(output_root, 'val')
    test = os.path.join(output_root, 'test')

    dataset = dict()
    dataset['info'] = dict()
    dataset['licenses'] = list()
    dataset['images'] = list()
    dataset['annotations'] = list()
    dataset['categories'] = list({'supercategory': 'person', 'id': 1, 'name': 'person'})

    img_id = 0
    ann_id = 0
    coco_idx = [i for i in range(len(coccp.ids))]
    for num in tqdm.tqdm(range(pow(len(coccp.ids), 2))):
        copy_idx = np.random.choice(coco_idx, 2, replace=False)

        image = coccp.load_example(copy_idx[0])

        if len(image['bboxes']) >= 3:
            continue

        paste = coccp.load_example(copy_idx[1])

        #select min mask
        area = 0
        for xmin, ymin, w, h, c, idx in image['bboxes']:
            a = w * h
            area = a if area < a else area

        max_area = area

        sidx = -1
        for xmin, ymin, w, h, c, idx in paste['bboxes']:
            a = w*h
            if area > a:
                sidx = idx
                area = a

        if sidx < 0:
            continue

        paste['masks'] = [paste['masks'][sidx]]
        xmin, ymin, w, h, c, i = paste['bboxes'][sidx]
        paste['bboxes'] = [[xmin, ymin, w, h, c, 0]]

        h, w, c = image['image'].shape
        ph, pw, pc = paste['image'].shape

        ch = min(h, ph)
        cw = min(w, pw)

        p = ch / max(h, ph)
        p1 = cw / max(w, pw)
        transforms = A.Compose([
            RandomCrop(ch, cw, no_random=p >= 0.8 and p1 >= 0.8),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=1.0, p=1)
            ],
            bbox_params=A.BboxParams(format="coco")
        )

        # src_image = image['image'].copy()
        # draw(src_image, image['masks'], image['bboxes'])
        # src_paste = paste['image'].copy()
        # draw(src_paste, paste['masks'], paste['bboxes'])

        output = transforms(
            image=image['image'], masks=image['masks'], bboxes=image['bboxes'],
            paste_image=paste['image'], paste_masks=paste['masks'], paste_bboxes=paste['bboxes']
        )

        new_img_name = '{}-{}-{}'.format(num, image['img_name'][:image['img_name'].rfind('.jpg')], paste['img_name'])
        cv2.imwrite(os.path.join(train, new_img_name), output['image'])
        coco_images_dict = dict()
        coco_images_dict['license'] = 1
        coco_images_dict['file_name'] = new_img_name
        coco_images_dict['flickr_url'] = coco_images_dict['coc_url'] = 'http://xintu.com/baoyong/{}'.format(new_img_name)
        coco_images_dict['height'] = output['image'].shape[0]
        coco_images_dict['width'] = output['image'].shape[1]
        coco_images_dict['date_captured'] = time.strftime('%Y-%m-%d %H:%M:%S')
        coco_images_dict['id'] = img_id
        dataset['images'].append(coco_images_dict)
        img_id += 1

        coco_annotations_list = list()
        for mask, (xmin, ymin, w, h, category_id, idx) in zip(output['masks'], output['bboxes']):
            coco_annotation_dict = dict()
            coco_annotation_dict['category_id'] = category_id
            coco_annotation_dict['image_id'] = coco_images_dict['id']
            coco_annotation_dict['id'] = ann_id
            coco_annotation_dict['iscrowd'] = 0
            coco_annotation_dict['bbox'] = [xmin, ymin, w, h]
            coco_annotation_dict['area'] = w * h
            plygons, holes = mask_to_polygons(mask)
            coco_annotation_dict['segmentation'] = [p.tolist() for p in plygons]
            coco_annotations_list.append(coco_annotation_dict)
            dataset['annotations'].extend(coco_annotations_list)
            ann_id += 1

        with open(os.path.join(annotations, 'instances_train_bck.json'), 'w') as w:
            json.dump(dataset, w)

        # draw_anno(output['image'], coco_annotations_list)
        #
        # #draw(output['image'], output['masks'], output['bboxes'])
        #
        # cv2.imshow('target', output['image'])
        # # cv2.imshow('src_paste', src_paste)
        # # cv2.imshow('src_image', src_image)
        # #
        # cv2.waitKey(3)

    with open(os.path.join(annotations, 'instances_train.json'), 'w') as w:
        json.dump(dataset, w)


