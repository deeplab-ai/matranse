# -*- coding: utf-8 -*-
"""Functions to load from and save to files."""

from __future__ import division

import json

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


def load_image(filename):
    """Read an image as numpy array."""
    img = Image.open(filename)
    img.load()
    return np.asarray(img, dtype="int32")


def save_image(nparray, filename):
    """Save a numpy array to an image file."""
    img = Image.fromarray(nparray.astype('uint8'))
    img.save(filename, mode='RGB')


def compute_relationship_probabilities(json_path='json_annos/'):
    """
    Compute probabilities P(Pred|<Sub,Obj>) from dataset.

    The Laplacian estimation is used:
        P(A|B) = (N(A,B)+1) / (N(B)+V_A),
    where:
        N(X) is the number of occurences of X and
        V_A is the number of different values that A can have

    Outputs:
        - sub_pred_obj_counts: 2-d dict of list of tuples
            (str, float)
            sub_pred_obj_counts[sub][obj] = [
                (pred, P(pred|<sub,obj>))
                for pred in predicates
            ]
    """
    # Load relationships and object and predicate names
    with open(json_path + 'annotation_train.json') as fid:
        annotations = json.load(fid)
    relationships = [
        (rel['subject'], rel['predicate'], rel['object'])
        for anno in annotations
        for rel in anno['relationships']
    ]
    with open(json_path + 'obj2vec.json') as fid:
        object_names = list(json.load(fid).keys())
    with open(json_path + 'predicate.json') as fid:
        predicate_names = json.load(fid)

    # Keep tracks of <Sub, Obj> pairs
    sub_obj_counts = {
        subj_name: {obj_name: 0 for obj_name in object_names}
        for subj_name in object_names
    }
    for subj, _, obj in relationships:
        sub_obj_counts[subj][obj] += 1

    # Keep tracks of <Sub, Pred, Obj>
    sub_pred_obj_counts = {
        subj_name: {
            obj_name: {pred: 1 for pred in predicate_names}
            for obj_name in object_names
        }
        for subj_name in object_names
    }
    for subj, pred, obj in relationships:
        sub_pred_obj_counts[subj][obj][pred] += 1

    # Probability of P|<S,O> (N(<S,P,O>)+1 / N(<S,O>)+N_preds)
    num_of_preds = len(predicate_names)
    for subj in object_names:
        for obj in object_names:
            sub_pred_obj_counts[subj][obj] = np.array([
                sub_pred_obj_counts[subj][obj][pred]
                / (sub_obj_counts[subj][obj] + num_of_preds)
                for pred in sorted(sub_pred_obj_counts[subj][obj].keys())
            ])
    return sub_pred_obj_counts


def load_annotations(mode, json_path='json_annos/'):
    """
    Load VRD annotations depending on mode.

    Inputs:
        - mode: str, either 'train', test', 'relationship', 'seen'
            or 'unseen'
        - json_path: str, the path where json annotations are stored
    Outputs:
        - annotations: list of dicts like the following
            {
                'filename': filename (contains no path),
                'relationships': [
                    {
                        'subject': str, subject_name,
                        'subject_box': [y_min, y_max, x_min, x_max],
                        'subject_id': int in [0, 99],
                        'subject_score': 1.0,
                        'predicate': str, predicate_name,
                        'predicate_id': int in [0, 69],
                        'object': object_name,
                        'object_box': [y_min, y_max, x_min, x_max],
                        'object_id': int in [0, 99],
                        'object_score': 1.0,
                        'id': int, this relationship's id,
                        'filename': filename + _id + extension
                    }
                ]
            }
    """
    if mode == 'relationship':
        with open(json_path + 'annotation_train.json', 'r') as fid:
            annotations = json.load(fid)
        with open(json_path + 'annotation_test.json', 'r') as fid:
            annotations += json.load(fid)
    elif mode in ('train', 'test'):
        with open(json_path + 'annotation_' + mode + '.json', 'r') as fid:
            annotations = json.load(fid)
    elif mode in ('unseen', 'seen'):
        with open(json_path + 'annotation_train.json', 'r') as fid:
            train_annotations = json.load(fid)
        with open(json_path + 'annotation_test.json', 'r') as fid:
            test_annotations = json.load(fid)
        train_annotations = set(
            (rel['subject'], rel['predicate'], rel['object'])
            for anno in train_annotations for rel in anno['relationships']
        )
        if mode == 'unseen':
            annotations = [
                {
                    'filename': anno['filename'],
                    'relationships': [
                        rel for rel in anno['relationships']
                        if (rel['subject'], rel['predicate'], rel['object'])
                        not in train_annotations
                    ]
                }
                for anno in test_annotations
            ]
        else:
            annotations = [
                {
                    'filename': anno['filename'],
                    'relationships': [
                        rel for rel in anno['relationships']
                        if (rel['subject'], rel['predicate'], rel['object'])
                        in train_annotations
                    ]
                }
                for anno in test_annotations
            ]
    return annotations


def torch_var(array, tensor_type=torch.FloatTensor, cuda_enabled=False):
    """Transform a numpy into a torch variable."""
    if not cuda_enabled:
        return Variable(torch.from_numpy(array)).type(tensor_type)
    return Variable(torch.from_numpy(array)).type(tensor_type).cuda()
