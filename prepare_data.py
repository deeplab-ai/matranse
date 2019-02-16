# -*- coding: utf-8 -*-
"""Functions to transform annotations and create train/test dataset."""

import os

import yaml

from src.annotation_transformer_class import AnnotationTransformer
from src.image_creator_class import ImageCreator
from src.mask_creator_class import MaskCreator
from src.resnet_feature_extractor import FeatureExtractor

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


def main(mode):
    """Run the data preprocessing and creation pipeline."""
    if not os.path.exists(CONFIG['models_path']):
        os.mkdir(CONFIG['models_path'])
    if not os.path.exists(CONFIG['figures_path']):
        os.mkdir(CONFIG['figures_path'])
    AnnotationTransformer().create_dataset_jsons()
    ImageCreator(mode=mode).create_dataset()
    MaskCreator(mode=mode).create_dataset()
    FeatureExtractor(mode=mode, use_cuda=CONFIG['use_cuda']).create_dataset()

if __name__ == "__main__":
    main('relationship')
