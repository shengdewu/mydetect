import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction
_PREDEFINED_SPLITS_COCO = {
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

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="/mnt/data/coco.data"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata('coco'),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    # for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
    #     # Assume pre-defined datasets live in `./datasets`.
    #     register_text_instances(
    #         key,
    #         metadata_text,
    #         os.path.join(root, json_file) if "://" not in json_file else json_file,
    #         os.path.join(root, image_root),
    #     )


# register_all_coco()