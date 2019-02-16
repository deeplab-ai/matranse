# -*- coding: utf-8 -*-
"""Create the object masks for our experiments."""

import math
import os

import numpy as np
import yaml

from src.utils.file_utils import load_image, load_annotations

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class MaskCreator():
    """
    Create object masks for feature extraction.

    Initialize an instance of this class (optionally) providing:
        - mode: str, either 'relationship', 'pair' or 'detected_pair'
            choose: 'relationship' for all relationships annotated
                    'pair' for all possible pairs, even not annotated
                    'detected_pair' for detected pairs
    """

    def __init__(self, mode='relationship'):
        """Iniitalize model parameters and load annotations."""
        self._mode = mode
        self._created_masks_path = CONFIG['created_masks_path']
        self._orig_images_path = CONFIG['orig_images_path']
        if not os.path.exists(self._created_masks_path):
            os.mkdir(self._created_masks_path)
        self._annotations = load_annotations(mode)

    def create_masks(self, mask_path, _create_mask):
        """
        Create masks for each relationship.

        For each image with name image_name and R relationships, R
        masks (arrays 2x32x32) are created, named image_name_r(type),
        where r in 0, ..., R-1. (image.jpg -> image_1.npy)

        Inputs:
            - mask_path: str, the path to store the mask arrays
            - _create_mask: function to create one mask array
        Creates:
            - A folder, if necessary
            - The files inside this folder
        """
        print('Creating files under ' + mask_path)
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        for img in self._annotations:
            all_masks_created = all(
                os.path.exists(self._created_mask_name(mask_path, rel))
                for rel in img['relationships']
            )
            img_filename = self._orig_images_path + img['filename']
            if os.path.exists(img_filename) and not all_masks_created:
                image = load_image(img_filename)
                for rel in img['relationships']:
                    filename = self._created_mask_name(mask_path, rel)
                    if not os.path.exists(filename):
                        np.save(filename, _create_mask(image, rel))
        print('Done')

    def create_dataset(self):
        """Create images for feature extraction."""
        folder_funcs = {
            self._mode + '_binary_masks/': self._binary_mask,
        }
        for folder, func in folder_funcs.items():
            self.create_masks(self._created_masks_path + folder, func)

    @staticmethod
    def _created_mask_name(mask_path, rel):
        # image.jpg -> mask_path/image_1.npy
        return mask_path + rel['filename'].split('.')[0] + '.npy'

    @staticmethod
    def _binary_mask(image, rel):
        mask = np.zeros((2, 32, 32))
        for btp, box_type in enumerate(['subject_box', 'object_box']):
            bbox = rel[box_type]
            height_ratio = 32.0 / image.shape[0]
            width_ratio = 32.0 / image.shape[1]
            y_min = max(0, int(math.floor(bbox[0] * height_ratio)))
            y_max = min(32, int(math.ceil(bbox[1] * height_ratio)))
            x_min = max(0, int(math.floor(bbox[2] * width_ratio)))
            x_max = min(32, int(math.ceil(bbox[3] * width_ratio)))
            mask[btp, y_min:y_max, x_min:x_max] = 1.0
            assert mask[btp].sum() == (y_max - y_min) * (x_max - x_min)
        return mask
