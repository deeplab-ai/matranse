# -*- coding: utf-8 -*-
"""Extract features with PyTorch ResNet-101."""

import os

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class ResNet101Pool5(nn.Module):
    """Pool 5 features with ResNet-101 PyTorch."""

    def __init__(self):
        """Initialize network and freeze layers."""
        super().__init__()
        self.resnet = nn.Sequential(
            *list(models.resnet101(pretrained=True).children())[:-1]
        )
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.linear = list(models.resnet101(pretrained=True).children())[-1]

    def forward(self, x):
        """Forward pass, return tensor (batch, 2048)."""
        return self.resnet(x).view(x.shape[0], -1)


class ResNet101Conv5(nn.Module):
    """Conv 5 features with ResNet-101 PyTorch."""

    def __init__(self):
        """Initialize network and freeze layers."""
        super().__init__()
        self.resnet = nn.Sequential(
            *list(models.resnet101(pretrained=True).children())[:-2]
        )
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass, return tensor (batch, 2048, 7, 7)."""
        return self.resnet(x)


def resnet_preprocessing(imgs, use_cuda):
    """
    Preprocess a list of images.

    Inputs:
        - imgs: list of Pillow images
        - use_cuda: bool, whether to load data on gpu
    Outputs:
        - tensor of preprocessed images
    """
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data = torch.stack([preprocessing(img) for img in imgs])
    if use_cuda:
        data = data.cuda()
    return data


class FeatureExtractor():
    """
    Perform feature extraction using ResNet features.

    Initialize an instance of this class (optionally) providing:
        - mode: str, either 'relationship', 'pair' or 'detected_pair'
            choose: 'relationship' for all relationships annotated
                    'pair' for all possible pairs, even not annotated
                    'detected_pair' for detected pairs
        - use_cuda: bool, whether to run on gpu mode
    """

    def __init__(self, mode='relationship', use_cuda=CONFIG['use_cuda']):
        """Initialize model parameters and load annotations."""
        self._mode = mode
        self._use_cuda = use_cuda
        self._created_images_path = CONFIG['created_images_path']
        self._features_path = CONFIG['features_path']
        if not os.path.exists(self._features_path):
            os.mkdir(self._features_path)
        self._resnet_conv5 = ResNet101Conv5().eval()
        self._resnet_pool5 = ResNet101Pool5().eval()
        if use_cuda:
            self._resnet_conv5 = self._resnet_conv5.cuda()
            self._resnet_pool5 = self._resnet_pool5.cuda()

    def create_features(self, features_path, images_path, resnet,
                        batch_size=32):
        """
        Create features for each relationship.

        Inputs:
            - features_path: str, the path to store the features
            - images_path: str, the path where the images are stored
            - resnet: nn.Module object
            - batch_size: int, the number of images per batch
        Creates:
            - A folder, if necessary
            - The images inside this folder
        """
        print('Creating files under ' + features_path)
        if not os.path.exists(features_path):
            os.mkdir(features_path)
        images = os.listdir(images_path)
        batch_starts = [
            ind for ind in range(len(images)) if ind % batch_size == 0
        ]
        for start in batch_starts:
            all_features_created = all(
                os.path.exists(self._created_feature_name(features_path, name))
                for name in images[start:start + batch_size]
            )
            if not all_features_created:
                features = resnet(resnet_preprocessing([
                    Image.open(images_path + img)
                    for img in images[start:start + batch_size]
                ], self._use_cuda)).cpu().detach().numpy()
                for cnt, name in enumerate(images[start:start + batch_size]):
                    np.save(
                        self._created_feature_name(features_path, name),
                        features[cnt]
                    )
        print('Done')

    def create_dataset(self):
        """Create images for feature extraction."""
        image_paths = [
            self._created_images_path + self._mode + '_union_boxes/',
            self._created_images_path + self._mode + '_union_boxes/',
            self._created_images_path + self._mode + '_subject_boxes/',
            self._created_images_path + self._mode + '_object_boxes/'
        ]
        feature_paths = [
            self._features_path + self._mode + '_union_boxes_conv5/',
            self._features_path + self._mode + '_union_boxes_pool5/',
            self._features_path + self._mode + '_subject_boxes_pool5/',
            self._features_path + self._mode + '_object_boxes_pool5/'
        ]
        funcs = [
            self._resnet_conv5, self._resnet_pool5,
            self._resnet_pool5, self._resnet_pool5
        ]
        for im_path, feat_path, func in zip(image_paths, feature_paths, funcs):
            self.create_features(feat_path, im_path, func)

    @staticmethod
    def _created_feature_name(features_path, name):
        # image_path/image_1.jpg -> features_path/image_1.npy
        return features_path + name.split('/')[-1].split('.')[0] + '.npy'
