# -*- coding: utf-8 -*-
"""Create the ancillary images for our experiments."""

import os

import numpy as np
import yaml

from src.utils.file_utils import load_image, save_image, load_annotations

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class ImageCreator():
    """
    Create images for feature extraction.

    Initialize an instance of this class (optionally) providing:
        - mode: str, either 'relationship', 'pair' or 'detected_pair'
            choose: 'relationship' for all relationships annotated
                    'pair' for all possible pairs, even not annotated
                    'detected_pair' for detected pairs
    """

    def __init__(self, mode='relationship'):
        """Initialize model parameters and load annotations."""
        self._mode = mode
        self._created_images_path = CONFIG['created_images_path']
        self._orig_images_path = CONFIG['orig_images_path']
        if not os.path.exists(self._created_images_path):
            os.mkdir(self._created_images_path)
        self._annotations = load_annotations(mode)

    def create_images(self, image_path, _create_image):
        """
        Create images for each relationship.

        For each image with name image_name and R relationships, R
        images are created, named image_name_r(type), where
        r in 0, ..., R-1. (image.jpg -> image_1.jpg)

        Inputs:
            - image_path: str, the path to store the images
            - _create_image: function to create one image
        Creates:
            - A folder, if necessary
            - The images inside this folder
        """
        print('Creating files under ' + image_path)
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        for img in self._annotations:
            all_images_created = all(
                os.path.exists(image_path + rel['filename'])
                for rel in img['relationships']
            )
            img_filename = self._orig_images_path + img['filename']
            if os.path.exists(img_filename) and not all_images_created:
                image = load_image(img_filename)
                for rel in img['relationships']:
                    filename = image_path + rel['filename']
                    if not os.path.exists(filename):
                        save_image(_create_image(image, rel), filename)
        print('Done')

    def create_dataset(self):
        """Create images for feature extraction."""
        folder_funcs = {
            self._mode + '_union_boxes/': self._boxes_union,
            self._mode + '_subject_boxes/': self._subject_box,
            self._mode + '_object_boxes/': self._object_box
        }
        for folder, func in folder_funcs.items():
            self.create_images(self._created_images_path + folder, func)

    @staticmethod
    def _boxes_union(image, rel):
        bbox = [
            min(rel['subject_box'][0], rel['object_box'][0]),
            max(rel['subject_box'][1], rel['object_box'][1]),
            min(rel['subject_box'][2], rel['object_box'][2]),
            max(rel['subject_box'][3], rel['object_box'][3])
        ]
        return image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]

    @staticmethod
    def _object_box(image, rel):
        bbox = rel['object_box']
        return image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]

    @staticmethod
    def _subject_box(image, rel):
        bbox = rel['subject_box']
        return image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
