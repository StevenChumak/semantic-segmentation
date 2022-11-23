import glob
import os
import pathlib
import random

import numpy as np
from PIL import Image
from runx.logx import logx

import datasets.uniform as uniform
import transforms
from config import cfg
from datasets.base_loader import BaseLoader


class Loader(BaseLoader):
    ignore_label = 255
    num_classes = 4
    trainid_to_name = {}
    color_mapping = []

    def __init__(
        self,
        mode,
        quality="semantic",
        joint_transform_list=None,
        img_transform=None,
        label_transform=None,
        eval_folder=None,
        albumentations=None,
        bg_swap=False,
    ):

        super(Loader, self).__init__(
            quality=quality,
            mode=mode,
            joint_transform_list=joint_transform_list,
            img_transform=img_transform,
            label_transform=label_transform,
            albumentations=albumentations,
        )
        
        
        self.labels = {
            ####### ego trackbed-rails single, neighbors combined #######
            "background": {"id": 0, "trainId": 0},
            "left_trackbed": {"id": 48, "trainId": 3},
            "left_rails": {"id": 49, "trainId": 3},
            "ego_trackbed": {"id": 50, "trainId": 1},
            "ego_rails": {"id": 51, "trainId": 2},
            "right_trackbed": {"id": 52, "trainId": 3},
            "right_rail": {"id": 53, "trainId": 3},
        }
        logx.msg(f"Using following Labeling: {self.labels}")

        self.trainid_to_name = {
            self.labels[label]["trainId"]: label for label in self.labels
        }
        self.id_to_trainid = {
            self.labels[label]["id"]: self.labels[label]["trainId"]
            for label in self.labels
        }

        self.fill_colormap()

        splits = {
            "train": "train",
            "val": "val",
            "folder": "test",
        }

        split_name = splits[mode]
        img_ext = "png"
        mask_ext = "png"

        # currently not in use due to not having a test set
        if eval_folder:
            img_root = os.path.join(eval_folder, "images")
            mask_root = os.path.join(eval_folder, "masks")
        else:
            self.root = cfg.DATASET.TRAINRAILS_DIR
            # self.root = cfg.DATASET.KNOTS_DIR
            extra = None
            size = str(cfg.DATASET.CROP_SIZE).split(",")
            
            size_str = "{}-{}".format(size[1], size[0])
            self.root = os.path.join(
                self.root, size_str, extra if extra is not None else ""
            )
            # self.root = os.path.join(self.root, extra if extra is not None else "")
            img_root = os.path.join(self.root, split_name, "images")
            mask_root = os.path.join(self.root, split_name, "masks")

        self.all_imgs = self.find_images(img_root, mask_root, img_ext, mask_ext)

        if mode == "train" and bg_swap:
            print(f"AYYYYY SWAPPING DA BACKGROUNND FOR {mode}")
            self.bg = True
            background_dir = cfg.DATASET.BACKGROUND_DIR
            background_filetype = cfg.DATASET.BACKGROUND_FILETYPE
            # Collect available background images.
            self.background_paths = glob.glob(
                os.path.join(background_dir, f"*.{background_filetype}")
            )
        else:
            print(f"NNNNNAYYYYY NOT SWAPPING DA BACKGROUNND FOR {mode}")
            self.bg = False
            self.background_paths = None

        logx.msg(f"num_classes {self.num_classes}")
        self.centroids = uniform.build_centroids(
            self.all_imgs,
            self.num_classes,
            self.train,
            cv=cfg.DATASET.CV,
            id2trainid=self.id_to_trainid,
        )

        self.build_epoch()

    def do_transforms(self, img, mask, centroid, img_name, class_id, background):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0
        
        if self.bg:
            import transforms.RandomSwapBackground as swap

            img = swap.swap_background(img, mask=mask, background=background)

        if self.albumentations is not None:
            transformed = self.albumentations(image=np.array(img), mask=np.array(mask))
        
            img = Image.fromarray(transformed["image"])
            mask = Image.fromarray(transformed["mask"])
            
        if self.img_transform is not None:
            img = self.img_transform(img)

        if cfg.DATASET.DUMP_IMAGES:
            self.dump_images(img_name, mask, centroid, class_id, img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)
            
        return img, mask, scale_float

    def __getitem__(self, index):
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs[index]

        mask_out = (
            cfg.DATASET.MASK_OUT_CITYSCAPES
            and cfg.DATASET.CUSTOM_COARSE_PROB is not None
            and "refinement" in mask_path
        )

        img, mask, img_name = self.read_images(img_path, mask_path, mask_out=mask_out)

        if self.background_paths is not None and self.bg:
            background_path = random.choice(self.background_paths)
            background = Image.open(background_path)
        else:
            background = None

        ######################################################################
        # Thresholding is done when using coarse-labelled Cityscapes images
        ######################################################################
        if "refinement" in mask_path:

            mask = np.array(mask)
            prob_mask_path = mask_path.replace(".png", "_prob.png")
            # put it in 0 to 1
            prob_map = np.array(Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = prob_map < cfg.DATASET.CUSTOM_COARSE_PROB
            mask[prob_map_threshold] = cfg.DATASET.IGNORE_LABEL
            mask = Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(
            img, mask, centroid, img_name, class_id, background=background
        )

        return img, mask, img_name, scale_float

    def fill_colormap(self):
        palette = [
            0,
            0,
            0,  # background        | schwarrz
            137,
            49,
            239,  # left_trackbed     | Blue-Violet
            242,
            202,
            25,  # left_rails        | Jonquil oder auch gelb
            225,
            24,
            69,  # ego_trackbed      | Spanish Crimson
            0,
            87,
            233,  # ego_rails         | RYB Blue
            135,
            233,
            17,  # right_trackbed    | Alien Armpit oder grün
            255,
            0,
            189,  # right_rails        | Shocking Pink
        ]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette
