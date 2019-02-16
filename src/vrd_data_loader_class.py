# -*- coding: utf-8 -*-
"""A class for data loading for the task of VRD."""

import json
import random
import os

import numpy as np
import torch
import yaml

from src.utils.file_utils import (
    load_annotations, torch_var, compute_relationship_probabilities
)

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class VRDDataLoader():
    """
    Custom data loader for Visual Relationships.

    Inputs upon initialization:
        - test_mode: str, choose:
            'relationship' to test predicate classification
            'pair' to test predicate classification and pair filtering
            'detected_pair' to test on detected object pairs
        - batch_size: int, batch size during training/testing
    """

    def __init__(self, test_mode='relationship', batch_size=1):
        """Initialize loader on train mode."""
        self._mode = {'train': 'relationship', 'test': test_mode}
        self._batch_size = batch_size
        self._created_masks_path = CONFIG['created_masks_path']
        self._features_path = CONFIG['features_path']
        self._json_path = CONFIG['json_path']
        self._orig_image_path = CONFIG['orig_images_path']
        with open(self._json_path + 'obj2vec.json') as fid:
            self._obj2vec = json.load(fid)
        self._encodings = np.eye(100)
        self._cond_probs = compute_relationship_probabilities(self._json_path)
        self._reset('train')
        self._use_cuda = False

    def _reset(self, mode):
        """Reset loader."""
        self._mode['current'] = mode
        self._annotations = load_annotations(
            self._mode['current'], self._json_path)
        self._set_targets()
        self._set_boxes()
        self._set_labels()
        self._set_probabilities()
        self._set_subject_object_one_hot_encodings()
        self._set_subject_object_embeddings()
        self._files = list(self._labels.keys())
        self._epoch = -1

    def _set_targets(self):
        orig_img_names = os.listdir(self._orig_image_path)
        self._targets = {
            rel['filename'].split('.')[0]: rel['predicate_id']
            for anno in self._annotations
            if anno['filename'] in orig_img_names
            for rel in anno['relationships']
        }

    def _set_boxes(self):
        orig_img_names = os.listdir(self._orig_image_path)
        self._boxes = {
            rel['filename'].split('.')[0]:
                [rel['subject_box'], rel['object_box']]
            for anno in self._annotations
            if anno['filename'] in orig_img_names
            for rel in anno['relationships']
        }

    def _set_labels(self):
        orig_img_names = os.listdir(self._orig_image_path)
        self._labels = {
            rel['filename'].split('.')[0]: [
                rel['subject_score'], 0.0, rel['object_score'],
                rel['subject_id'], -1, rel['object_id']
            ]
            for anno in self._annotations
            if anno['filename'] in orig_img_names
            for rel in anno['relationships']
        }

    def _set_probabilities(self):
        self._probabilities = {
            rel['filename'].split('.')[0]:
                self._cond_probs[rel['subject']][rel['object']]
            for anno in self._annotations for rel in anno['relationships']
        }

    def _set_subject_object_one_hot_encodings(self):
        self._subject_encodings = {
            rel['filename'].split('.')[0]: self._encodings[rel['subject_id']]
            for anno in self._annotations for rel in anno['relationships']
        }
        self._object_encodings = {
            rel['filename'].split('.')[0]: self._encodings[rel['object_id']]
            for anno in self._annotations for rel in anno['relationships']
        }

    def _set_subject_object_embeddings(self):
        self._subject_embeddings = {
            rel['filename'].split('.')[0]: self.obj2vec(rel['subject'])
            for anno in self._annotations for rel in anno['relationships']
        }
        self._object_embeddings = {
            rel['filename'].split('.')[0]: self.obj2vec(rel['object'])
            for anno in self._annotations for rel in anno['relationships']
        }

    def _check_epoch(self, epoch):
        """If new epoch detected, shuffle images."""
        if epoch > self._epoch:
            self._epoch += 1
            random.shuffle(self._files)

    def get_batches(self):
        """Get the start index of all batches."""
        return [
            b for b in range(len(self._files)) if b % self._batch_size == 0
        ]

    def get_probabilities(self, epoch, batch):
        """Return tensors of probabilities (70,) for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._probabilities[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_one_hot_subject_encodings(self, epoch, batch):
        """Return tensors of embeddings (300,) for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._subject_encodings[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_one_hot_object_encodings(self, epoch, batch):
        """Return tensors of embeddings (300,) for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._object_encodings[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_subject_embeddings(self, epoch, batch):
        """Return tensors of embeddings (300,) for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._subject_embeddings[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_object_embeddings(self, epoch, batch):
        """Return tensors of embeddings (300,) for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._object_embeddings[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_union_boxes_pool5_features(self, epoch, batch):
        """Return a tensor of features for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            np.load(
                self._features_path + self._mode[self._mode['current']]
                + '_union_boxes_pool5/' + name + '.npy'
            )
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_subject_boxes_pool5_features(self, epoch, batch):
        """Return a tensor of features for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            np.load(
                self._features_path + self._mode[self._mode['current']]
                + '_subject_boxes_pool5/' + name + '.npy'
            )
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_object_boxes_pool5_features(self, epoch, batch):
        """Return a tensor of features for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            np.load(
                self._features_path + self._mode[self._mode['current']]
                + '_object_boxes_pool5/' + name + '.npy'
            )
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_union_boxes_conv5_features(self, epoch, batch):
        """Return a tensor of features for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            np.load(
                self._features_path + self._mode[self._mode['current']]
                + '_union_boxes_conv5/' + name + '.npy'
            )
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_masks(self, epoch, batch):
        """Return a tensor of features for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            np.load(
                self._created_masks_path + self._mode[self._mode['current']]
                + '_binary_masks/' + name + '.npy'
            )
            for name in self._files[batch: batch + self._batch_size]
        ]), cuda_enabled=self._use_cuda)

    def get_targets(self, epoch, batch):
        """Return a tensor of predicate targets for current batch."""
        self._check_epoch(epoch)
        return torch_var(np.array([
            self._targets[name]
            for name in self._files[batch: batch + self._batch_size]
        ]), tensor_type=torch.LongTensor, cuda_enabled=self._use_cuda)

    def get_boxes(self, epoch, batch):
        """Return a dict of bounding boxes for current batch."""
        self._check_epoch(epoch)
        return {
            name: self._boxes[name]
            for name in self._files[batch: batch + self._batch_size]
        }

    def get_labels(self, epoch, batch):
        """Return a dict of labels for current batch."""
        self._check_epoch(epoch)
        return {
            name: self._labels[name]
            for name in self._files[batch: batch + self._batch_size]
        }

    def get_files(self, epoch, batch):
        """Return the file names of current batch."""
        self._check_epoch(epoch)
        return list(self._files[batch: batch + self._batch_size])

    def obj2vec(self, object_tag):
        """Return the embedding of an object tag."""
        return np.array(self._obj2vec[object_tag]).flatten()

    def train(self, batch_size=None):
        """Set instance to train mode, return self."""
        self._reset('train')
        if batch_size is not None:
            self._batch_size = batch_size
        return self

    def eval(self, batch_size=None):
        """Set instance to eval mode, return self."""
        self._reset('test')
        if batch_size is not None:
            self._batch_size = batch_size
        return self

    def cpu(self):
        """Set cpu mode on."""
        self._use_cuda = False
        return self

    def cuda(self):
        """Set gpu mode on."""
        self._use_cuda = True
        return self
