# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiered Subset of ImageNet Dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pickle
import numpy as np
from tqdm import tqdm


from absl import logging
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{ren18fewshotssl,
author   = {Mengye Ren and
          Eleni Triantafillou and
          Sachin Ravi and
          Jake Snell and
          Kevin Swersky and
          Joshua B. Tenenbaum and
          Hugo Larochelle and
          Richard S. Zemel},
title    = {Meta-Learning for Semi-Supervised Few-Shot Classification},
booktitle= {Proceedings of 6th International Conference on Learning Representations {ICLR}},
year     = {2018},
}
"""
_DESCRIPTION = """\
The tieredImageNet is a dataset proposed specifically for few-shot classification.
Like miniImagenet, it is subset of ILSVRC-12, but larger subset with total 608 classes.
The dataset is proposed to make use of class hierarchy while few-shots classficicatio.
It is analogous to Omniglot dataset where the characters are grouped into alphabets.
tieredImageNet groups classes into broader categories that corresponds to higher-level nodes
in the ImageNet hierarchy.
tieredImageNet has total 34 general categories. Each category containing between 10 and 30
classes. The categories are split into 20 training, 6 validation and 8 testing categories.
Each category has corresponding more specific classes.
"""

_BASE_URL = "https://arxiv.org/pdf/1803.00676.pdf"
# _DL_URL = "https://drive.google.com/uc?export=download&id=1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u"
_DL_URL = "https://drive.google.com/uc?export=download&id=1tPDwzBpKlxMT4-Jkyw2OpHsLPjPyh4LD"


class TieredImagenet(tfds.core.GeneratorBasedBuilder):
    """tiered_imagenet is comparatively larger subset of Imagenet that uses splits defined for benchmarking few-shot learning approaches"""
    VERSION = tfds.core.Version("1.0.2")
    _IMAGE_SIZE_X = 84
    _IMAGE_SIZE_Y = 84
    _NUM_CLASSES = 608
    _NUM_CLASSES_TRAIN = 351
    _NUM_CLASSES_VALIDATION = 97
    _NUM_CLASSES_TEST = 160
    _IMAGE_SHAPE = (_IMAGE_SIZE_X, _IMAGE_SIZE_Y, 3)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "label": tfds.features.ClassLabel(num_classes=self._NUM_CLASSES),
            }),
            supervised_keys=("image", "label"),
            urls=[_BASE_URL],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Downloads the data and defines the split."""
        extracted_path = dl_manager.download_and_extract(_DL_URL)
        logging.info("extracted path %s", extracted_path)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=5,
                gen_kwargs={
                    "images_path": os.path.join(str(extracted_path), "train_images_png.pkl"),
                    "labels_path": os.path.join(str(extracted_path), "train_labels.pkl"),
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=5,
                gen_kwargs={
                    "images_path": os.path.join(str(extracted_path), "test_images_png.pkl"),
                    "labels_path": os.path.join(str(extracted_path), "test_labels.pkl"),
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=5,
                gen_kwargs={
                    "images_path": os.path.join(str(extracted_path), "val_images_png.pkl"),
                    "labels_path": os.path.join(str(extracted_path), "val_labels.pkl"),
                }
            ),

        ]

    def _generate_examples(self, images_path, labels_path):
        """yields examples from the dataset.
        Args:
            images_path: path to extracted images.
            labels_path: path to extracted labels.
        Yields:
                keys:
                    "image" (np.ndarray, shape=[84, 84, 3],
                                  dtype=np.uint8)
                    "label" (string): class name.
        """
        # read data
        with open(labels_path, "rb") as labels_file:
            data = pickle.load(labels_file, encoding='latin1')
            list_label_specific = data["label_specific"]
            logging.info("Labels list %s", list_label_specific)
        with tf.gfile.GFile(images_path, "rb") as img_file:
            img_array = pickle.load(img_file, encoding="latin1")
            img_data = np.zeros(
                [len(img_array["images"]), self._IMAGE_SIZE_X, self._IMAGE_SIZE_Y, 3], dtype=np.uint8)
            for img_index, item in tqdm(enumerate(img_array["images"])):
                img_data[img_index] = item
        for i in range(0, len(img_array["images"])):
            dict_data = {
                "image": img_data[i],
                "label": list_label_specific[i]
            }
        yield dict_data
