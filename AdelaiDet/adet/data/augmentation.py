import random

import numpy as np
from fvcore.transforms import transform as T

from detectron2.data.transforms import RandomCrop, StandardAugInput, RandomSaturation, RandomContrast, RandomBrightness
from detectron2.data.transforms.augmentation import Augmentation
from adet.data.transform import ColorTransform
import logging


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return T.CropTransform(0, 0, image_size[1], image_size[0])

    return T.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class RandomColorAugmentation(Augmentation):
    def __init__(self, factor=None):
        self.factor = factor
        return

    def get_transform(self, image):
        return ColorTransform(self.factor)


class RandomAugmentation:

    def __init__(self, cfg, prob=0.5):
        self.aug = self.__build_random_augmentation(cfg)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        return

    @staticmethod
    def __build_random_augmentation(cfg):
        random_augmentation = list()

        if cfg.INPUT.CROP.ENABLED:
            random_augmentation.append(
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                )
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(random_augmentation[-1])
            )

        if cfg.INPUT.BRIGHT.ENABLED:
            random_augmentation.append(
                RandomBrightness(
                    cfg.INPUT.BRIGHT.MIN,
                    cfg.INPUT.BRIGHT.MAX
                )
            )
            logging.getLogger(__name__).info(
                "Brightness used in training: " + str(random_augmentation[-1])
            )

        if cfg.INPUT.CONTRAST.ENABLED:
            random_augmentation.append(
                RandomContrast(
                    cfg.INPUT.CONTRAST.MIN,
                    cfg.INPUT.CONTRAST.MAX
                )
            )
            logging.getLogger(__name__).info(
                "Contrast used in training: " + str(random_augmentation[-1])
            )

        if cfg.INPUT.SATURATION.ENABLED:
            random_augmentation.append(
                RandomSaturation(
                    cfg.INPUT.SATURATION.MIN,
                    cfg.INPUT.SATURATION.MAX
                )
            )

            logging.getLogger(__name__).info(
                "Saturation used in training: " + str(random_augmentation[-1])
            )

        if cfg.INPUT.COLOR.ENABLED:
            random_augmentation.append(
                RandomColorAugmentation()
            )

            logging.getLogger(__name__).info(
                "COLOR used in training: " + str(random_augmentation[-1])
            )
        return random_augmentation

    def __call__(self, base_aug):
        assert isinstance(base_aug, list), 'the base augmentation must be list[Augmentation]'
        do = len(self.aug) > 0 and np.random.random() < self.prob
        if do:
            augmentation = base_aug.copy()
            num = np.random.randint(1, len(self.aug))
            augs = np.random.choice(self.aug, num, replace=False) #replace: prevent duplication for np.random.choice
            for aug in augs:
                augmentation.insert(
                    0,
                    aug
                )
            return augmentation
        else:
            return base_aug
