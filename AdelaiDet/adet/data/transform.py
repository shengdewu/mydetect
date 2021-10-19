from fvcore.transforms.transform import (
    NoOpTransform,
    Transform,
)

import numpy as np

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

__all__ = [
    "ColorTransform",
]


class ColorTransform(Transform):
    """
     out = image1 * (1.0 - alpha) + image2 * alpha
     BGR
    """

    def __init__(self, factor):
        super().__init__()
        # self.blue = (0, 0, 255)
        # self.green = (0, 128, 0)
        # self.red = (255, 0, 0)
        self.mode = ['blue', 'green', 'red']
        self.factor = factor

        return

    def apply_image(self, img):
        mask = np.zeros_like(img)
        mode_id = np.random.randint(0, len(self.mode))
        mode = self.mode[mode_id]
        if mode == 'blue':
            mask[:, :, 0] = mask[:, :, 0] + 255
        elif mode == 'green':
            mask[:, :, 1] = mask[:, :, 1] + 128
        else:
            mask[:, :, 2] = mask[:, :, 2] + 255

        r = np.random.randint(low=0, high=51) / 100.0 if self.factor is None else self.factor

        return (img * (1.0 - r) + mask * r).astype(img.dtype)

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation
