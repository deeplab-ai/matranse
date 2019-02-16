# -*- coding: utf-8 -*-
"""Functions to compute different metrics for evaluation."""

from __future__ import division

import json

import numpy as np
from scipy.io import loadmat
from src.utils.file_utils import load_annotations


def compute_area(bbox):
    """Compute area of box 'bbox' ([y_min, y_max, x_min, x_max])."""
    return max(0, bbox[3] - bbox[2] + 1) * max(0, bbox[1] - bbox[0] + 1)


def compute_overlap(det_bboxes, gt_bboxes):
    """
    Compute overlap of detected and ground truth boxes.

    Inputs:
        - det_bboxes: array (2, 4), 2 x [y_min, y_max, x_min, x_max]
            The detected bounding boxes for subject and object
        - gt_bboxes: array (2, 4), 2 x [y_min, y_max, x_min, x_max]
            The ground truth bounding boxes for subject and object
    Returns:
        - overlap: non-negative float <= 1
    """
    overlaps = []
    for det_bbox, gt_bbox in zip(det_bboxes, gt_bboxes):
        intersection_bbox = [
            max(det_bbox[0], gt_bbox[0]),
            min(det_bbox[1], gt_bbox[1]),
            max(det_bbox[2], gt_bbox[2]),
            min(det_bbox[3], gt_bbox[3])
        ]
        intersection_area = compute_area(intersection_bbox)
        union_area = (compute_area(det_bbox)
                      + compute_area(gt_bbox)
                      - intersection_area)
        overlaps.append(intersection_area / union_area)
    return min(overlaps)


def relationship_recall(n_re, det_labels, det_bboxes, gt_labels,
                        gt_bboxes):
    """
    Evaluate relationship recall, with top n_re predictions per image.

    Inputs:
        - n_re: int, number of most confident predictions to keep
        - det_labels: list of num_imgs arrays (N, 6), where N is the
            number of predictions in this image. Each row follows
            the format: [
                subj_conf, pred_conf, obj_conf,
                subj_tag, pred_tag, obj_tag
            ]
            The detected labels and confidence scores
        - det_bboxes: list of num_imgs arrays (N, 2, 4), where N is the
            number of predictions in this image. Each 2x4 array follows
            the format: [
                [x_min_subj, y_min_subj, x_max_subj, y_max_subj]
                [x_min_obj, y_min_obj, x_max_obj, y_max_obj]
            ]
            The detected bounding boxes for subject and object
        - gt_labels: list of num_imgs arrays (N, 3), where N is the
            number of annotations in this image. Each row follows
            the format: [subj_tag, pred_tag, obj_tag]
            The ground truth labels
        - gt_bboxes: list of num_imgs arrays (N, 2, 4), where N is the
            number of annotations in this image. Each 2x4 array follows
            the format: [
                [x_min_subj, y_min_subj, x_max_subj, y_max_subj]
                [x_min_obj, y_min_obj, x_max_obj, y_max_obj]
            ]
            The ground truth bounding boxes for subject and object
    Returns:
        - the computed recall metric (non-negative float <= 1)
    """
    relationships_found = 0
    all_relationships = sum(labels.shape[0] for labels in gt_labels)
    for item in zip(det_labels, det_bboxes, gt_labels, gt_bboxes):
        (det_lbls, det_bxs, gt_lbls, gt_bxs) = item
        if not det_lbls.any() or not gt_lbls.any():
            continue  # omit empty detection matrices
        gt_detected = np.zeros(gt_lbls.shape[0])
        det_score = np.sum(np.log(det_lbls[:, 0:3]), axis=1)
        inds = np.argsort(det_score)[::-1][:n_re]  # at most n_re predictions
        for det_box, det_label in zip(det_bxs[inds, :], det_lbls[inds, 3:]):
            overlaps = np.array([
                max(compute_overlap(det_box, gt_box), 0.499)
                if detected == 0 and not any(det_label - gt_label)
                else 0
                for gt_box, gt_label, detected
                in zip(gt_bxs, gt_lbls, gt_detected)
            ])
            if (overlaps >= 0.5).any():
                gt_detected[np.argmax(overlaps)] = 1
                relationships_found += 1
    return relationships_found / all_relationships


def evaluate_relationship_recall(scores, boxes, labels, max_predictions,
                                 mode='relationship'):
    """
    Evaluate recall on the test data.

    Inputs:
        - scores: dict, {filename: list of scores}
        - boxes: dict, {filename: [subject_box, object_box]}
        - labels: dict, {
            filename: [subj_score, 0, obj_score, subj_tag, -1, obj_tag]
        }
        - max_predictions: int, maximum predictions to keep per pair,
        - mode: str, either 'relationship', 'seen' or 'unseen'
    Returns:
        - recall@50 (non-negative float <= 1) on test data
        - recall@100 (non-negative float <= 1) on test data
    """
    scores = {
        filename: list(zip(
            np.argsort(score_vec)[::-1], np.sort(score_vec)[::-1]
        ))[:max_predictions]
        for filename, score_vec in scores.items()
    }
    det_labels, det_bboxes = merge_predictions(scores, boxes, labels)
    gt_labels, gt_bboxes = gt_labels_and_bboxes(mode)
    det_labels.update(
        {name: np.array([]) for name in gt_labels if name not in det_labels}
    )
    det_bboxes.update(
        {name: np.array([]) for name in gt_bboxes if name not in det_bboxes}
    )

    # Align file names of tested and annotated images
    if len(det_labels.keys()) != len(gt_labels.keys()):
        raise ValueError('Found file names that are not annotated.')
    filenames = sorted(det_labels.keys())
    det_labels = [det_labels[name] for name in filenames]
    gt_labels = [gt_labels[name] for name in filenames]
    det_bboxes = [det_bboxes[name] for name in filenames]
    gt_bboxes = [gt_bboxes[name] for name in filenames]
    recall_50 = relationship_recall(
        50, det_labels, det_bboxes, gt_labels, gt_bboxes
    )
    recall_100 = relationship_recall(
        100, det_labels, det_bboxes, gt_labels, gt_bboxes
    )
    return recall_50, recall_100


def merge_predictions(scores, boxes, labels):
    """
    Merge detected scores, labels and bounding boxes to desired format.

    Inputs:
        - scores: dict, {filename: list of (labels, scores)}
        - boxes: dict, {filename: [subject_box, object_box]}
        - labels: dict, {
            filename: [subj_score, 0, obj_score, subj_tag, -1, obj_tag]
        }
    Returns:
        - det_labels: dict of lists of (N, 6) arrays, with rows like
            [subj_conf, pred_conf, obj_conf, subj_id, pred_id, obj_id]
        - det_bboxes: dict of lists of (N, 2, 4) arrays, with rows like
            [subject_box, object_box]

    """
    # Merge scores and labels
    labels = {
        name: [
            [label[0], pred_score, label[2], label[3], pred_id, label[5]]
            for pred_id, pred_score in scores[name]
        ]
        for name, label in labels.items()
    }

    # Align labels and boxes
    boxes = {
        name: [boxes_list for _ in range(len(labels[name]))]
        for name, boxes_list in boxes.items()
    }

    # Group detections based on original image name
    orig_filenames = set(name[:name.rfind('_')] for name in scores.keys())
    det_labels, det_bboxes = {}, {}
    for orig_name in orig_filenames:
        rel_names = [
            name for name in labels.keys()
            if name[:name.rfind('_')] == orig_name
        ]
        det_labels[orig_name] = np.array([
            label_list for name in rel_names for label_list in labels[name]
        ])
        det_bboxes[orig_name] = np.array([
            boxes_list for name in rel_names for boxes_list in boxes[name]
        ])
    return det_labels, det_bboxes


def gt_labels_and_bboxes(mode):
    """
    Return ground truth labels and bounding boxes.

    - gt_labels: dict of list of (N, 3) arrays (subj, pred, obj)
    - gt_bboxes: dict of list of (N, 2, 4) arrays (boxes)
    """
    if mode == 'relationship':
        annotations = load_annotations('test')
    else:
        annotations = load_annotations(mode)
    gt_labels = {
        anno['filename'][:anno['filename'].rfind('.')]: np.array([
            [rel['subject_id'], rel['predicate_id'], rel['object_id']]
            for rel in anno['relationships']
        ])
        for anno in annotations
    }
    gt_bboxes = {
        anno['filename'][:anno['filename'].rfind('.')]: np.array([
            [rel['subject_box'], rel['object_box']]
            for rel in anno['relationships']
        ])
        for anno in annotations
    }
    return gt_labels, gt_bboxes
