import os

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
import datasets.uniform as uniform
import glob
import random
from PIL import Image
import numpy as np

import transforms

class Loader(BaseLoader):
    num_classes = 3
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='semantic', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None, bg_swap=False):

        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)
                                     
        self.root = cfg.DATASET.RAILSEM19_DIR
        self.labels = {
            "road": {"id": 0, "trainId": 0, "color": (0, 0, 0)},
            "sidewalk": {"id": 1, "trainId": 0, "color": (0, 0, 0)},
            "construction": {"id": 2, "trainId": 0, "color": (0, 0, 0)},
            "tram-track": {"id": 3, "trainId": 0, "color": (0, 0, 0)},
            "fence": {"id": 4, "trainId": 0, "color": (0, 0, 0)},
            "pole": {"id": 5, "trainId": 0, "color": (0, 0, 0)},
            "traffic-light": {"id": 6, "trainId": 0, "color": (0, 0, 0)},
            "traffic-sign": {"id": 7, "trainId": 0, "color": (0, 0, 0)},
            "vegetation": {"id": 8, "trainId": 0, "color": (0, 0, 0)},
            "terrain": {"id": 9, "trainId": 0, "color": (0, 0, 0)},
            "sky": {"id": 10, "trainId": 0, "color": (0, 0, 0)},
            "human": {"id": 11, "trainId": 0, "color": (0, 0, 0)},

            "rail-track": {"id": 12, "trainId": 1, "color": (137, 49, 239)},

            "car": {"id": 13, "trainId": 0, "color": (0, 0, 0)},
            "truck": {"id": 14, "trainId": 0, "color": (0, 0, 0)},
            "trackbed": {"id": 15, "trainId": 0, "color": (0, 0, 0)},
            "on-rails": {"id": 16, "trainId": 0, "color": (0, 0, 0)},

            "rail-raised": {"id": 17, "trainId": 2, "color": (242, 202, 25)},

            "rail-embedded": {"id": 18, "trainId": 0, "color": (0, 0, 0)},
        }

        self.trainid_to_name   = { self.labels[label]["trainId"]   :   label for label in self.labels   }
        self.id_to_trainid   = { self.labels[label]["id"]   :   self.labels[label]["trainId"] for label in self.labels   }

        self.fill_colormap()

        splits = {'train': 'trn',
                    'val': 'val'
                 }
                 
        split_name = splits[mode]
        img_ext = 'png'
        mask_ext = 'png'
        img_root = os.path.join(self.root, 'images', split_name)
        mask_root = os.path.join(self.root, 'masks', split_name)

        self.all_imgs = self.find_images(img_root, mask_root, img_ext,
                                             mask_ext)

        logx.msg(f'cn num_classes {self.num_classes}')
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                self.num_classes,
                                                self.train,
                                                cv=cfg.DATASET.CV,
                                                id2trainid=self.id_to_trainid)

        self.build_epoch()

    def fill_colormap(self):
        palette = [
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  225, 24, 69,  # ego_trackbed      | Spanish Crimson
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0,0,0,        # background        | schwarrz
                  0, 87, 233,   # ego_rails         | RYB Blue
                  0,0,0,        # background        | schwarrz
                  ]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette