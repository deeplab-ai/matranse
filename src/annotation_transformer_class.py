# -*- coding: utf-8 -*-
"""
A class to transform the matlab annotations into python annotations.

Run create_dataset_jsons() to transform all .mat files to json files.
The json files that are to be created are:
    - annotation_train.json
    - annotation_test.json
    - obj2vec.json
    - predicate.json

See the corresponding methods for more info.
"""

import json
import os

import yaml
from scipy.io import loadmat

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class AnnotationTransformer():
    """Transform matlab annotations to json."""

    def __init__(self):
        """Initialize AnnotationTransformer."""
        self._json_path = CONFIG['json_path']
        self._mat_path = CONFIG['mat_path']
        if self._json_path and not os.path.exists(self._json_path):
            os.mkdir(self._json_path)

    def create_dataset_jsons(self):
        """
        Create jsons if they do not already exist in json_path.

        This method loads mat files from mat_path and transforms the
        matlab dataset into python annotations, saved in json_path.
        Files "annotation_train.json", "annotation_test.json",
        "obj2vec.json", "predicate.json" are created.
        """
        self._create_obj2vec_json()
        self._create_predicate_json()
        self._create_annotation_json('train')
        self._create_annotation_json('test')

    def _create_annotation_json(self, mode):
        """
        Load annotations from "annotation_*.mat" to "annotation_*.json".

        Inputs:
            - mode: str, either 'train' or 'test'
        Outputs:
            - list of dicts like
                {
                    'filename': filename (no path),
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
        with open(self._json_path + 'obj2vec.json') as fid:
            objects = {
                obj: o for o, obj in enumerate(sorted(json.load(fid).keys()))
            }
        with open(self._json_path + 'predicate.json') as fid:
            predicates = {
                pred: p for p, pred in enumerate(json.load(fid))
            }
        json_filename = self._json_path + 'annotation_' + mode + '.json'
        if not os.path.exists(json_filename):
            annos = loadmat(self._mat_path + 'annotation_' + mode + '.mat')
            annos = annos['annotation_' + mode][0]
            json_annos = [
                {
                    'filename': anno[0]['filename'][0][0],
                    'relationships': [
                        {
                            'subject': rel[0]['phrase'][0][0][0][0],
                            'subject_box': rel[0]['subBox'][0][0].tolist(),
                            'subject_id':
                                objects[rel[0]['phrase'][0][0][0][0]],
                            'subject_score': 1.0,
                            'predicate': rel[0]['phrase'][0][0][1][0],
                            'predicate_id':
                                predicates[rel[0]['phrase'][0][0][1][0]],
                            'object': rel[0]['phrase'][0][0][2][0],
                            'object_box': rel[0]['objBox'][0][0].tolist(),
                            'object_id': objects[rel[0]['phrase'][0][0][2][0]],
                            'object_score': 1.0,
                            'id': r,
                            'filename': (
                                anno[0]['filename'][0][0].split('.')[0]
                                + '_' + str(r)
                                + '.'
                                + anno[0]['filename'][0][0].split('.')[-1]
                            )
                        }
                        for r, rel in enumerate(anno[0]['relationship'][0][0])
                    ]
                }
                if self._handle_no_relationships(anno)
                else
                {
                    'filename': anno[0]['filename'][0][0],
                    'relationships': []
                }
                for anno in annos
            ]
            with open(json_filename, 'w') as fid:
                json.dump(json_annos, fid)

    def _create_obj2vec_json(self):
        """
        Load annotations from "cell_obj2vec.mat" to "obj2vec.json".

        The output file contains a dictionary that has the following
        structure:
        {
            obj_name_1: np.array 1x300,
            obj_name_2: np.array 1x300,
            .
            .
            .
            obj_name_N: np.array 1x300
        }
        where N the number of total objects (100 in VRD).
        """
        if not os.path.exists(self._json_path + 'obj2vec.json'):
            annos = loadmat(self._mat_path + 'cell_obj2vec.mat')
            json_annos = {
                anno[0][0][0]: anno[0][1][0].tolist()
                for anno in annos['cell_obj2vec'][0]
            }
            with open(self._json_path + 'obj2vec.json', 'w') as fid:
                json.dump(json_annos, fid)

    def _create_predicate_json(self):
        """
        Load annotations from "predicate.mat" to "predicate.json".

        The output file contains a list:
        [
            pred_name_1,
            .
            .
            .
            pred_name_N
        ]
        where N the number of total predicates (70 in VRD).
        """
        if not os.path.exists(self._json_path + 'predicate.json'):
            annos = loadmat(self._mat_path + 'predicate.mat')['predicate'][0]
            json_annos = sorted([anno[0] for anno in annos])
            with open(self._json_path + 'predicate.json', 'w') as fid:
                json.dump(json_annos, fid)

    @staticmethod
    def _handle_no_relationships(anno):
        """Check if annotation 'anno' has a relationship part."""
        try:
            anno[0]['relationship']
            return True
        except:
            return False

if __name__ == "__main__":
    AnnotationTransformer().create_dataset_jsons()
