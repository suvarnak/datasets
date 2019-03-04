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

"""Tests for tensorflow_datasets.core.dataset_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
from tensorflow_datasets import testing
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import dataset_info
from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import download
from tensorflow_datasets.core import features
from tensorflow_datasets.core import lazy_imports
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils


tf.compat.v1.enable_eager_execution()


class DummyBeamDataset(dataset_builder.BeamBasedBuilder):

  VERSION = utils.Version("1.0.0")

  def _info(self):

    return dataset_info.DatasetInfo(
        builder=self,
        features=features.FeaturesDict({
            "image": features.Image(shape=(16, 16, 1)),
            "label": features.ClassLabel(names=["dog", "cat"]),
        }),
        supervised_keys=("x", "x"),
    )

  def _split_generators(self, dl_manager):
    del dl_manager
    return [
        splits_lib.SplitGenerator(
            name=splits_lib.Split.TRAIN,
            num_shards=10,
            gen_kwargs=dict(num_examples=1000),
        ),
        splits_lib.SplitGenerator(
            name=splits_lib.Split.TEST,
            num_shards=4,
            gen_kwargs=dict(num_examples=725),
        ),
    ]

  def _build_pcollection(self, pipeline, num_examples):
    """Generate examples as dicts."""

    beam = lazy_imports.lazy_imports.apache_beam

    def _process_example(x):
      return {
          "image": np.ones((16, 16, 1), np.uint8),
          "label": x % 2,
      }

    return (
        pipeline
        | beam.Create(range(num_examples))
        | beam.Map(_process_example)
    )


class BeamBasedBuilderTest(testing.TestCase):

  def test_download_prepare_raise(self):
    with testing.tmp_dir(self.get_temp_dir()) as tmp_dir:
      builder = DummyBeamDataset(data_dir=tmp_dir)
      with self.assertRaisesWithPredicateMatch(ValueError, "no Beam Runner"):
        builder.download_and_prepare()

  def _assertBeamGeneration(self, dl_config):
    with testing.tmp_dir(self.get_temp_dir()) as tmp_dir:
      builder = DummyBeamDataset(data_dir=tmp_dir)
      builder.download_and_prepare(download_config=dl_config)

      data_dir = os.path.join(tmp_dir, "dummy_beam_dataset", "1.0.0")
      self.assertEqual(data_dir, builder._data_dir)

      # Check number of shards
      self._assertShards(
          data_dir,
          pattern="dummy_beam_dataset-test.tfrecord-{:05}-of-{:05}",
          num_shards=4,
      )
      self._assertShards(
          data_dir,
          pattern="dummy_beam_dataset-train.tfrecord-{:05}-of-{:05}",
          num_shards=10,
      )

      datasets = dataset_utils.as_numpy(builder.as_dataset())
      self.assertEqual(
          sorted(ex["label"] for ex in datasets["test"]),
          sorted([i % 2 for i in range(725)]),
      )

  def _assertShards(self, data_dir, pattern, num_shards):
    shards_filenames = [
        pattern.format(i, num_shards) for i in range(num_shards)
    ]
    self.assertTrue(all(
        tf.io.gfile.exists(os.path.join(data_dir, f)) for f in shards_filenames
    ))


  if not six.PY2:

    # The default beam pipeline do not works with Python2
    def test_download_prepare(self):
      dl_config = download.DownloadConfig(
          beam_options=beam.options.pipeline_options.PipelineOptions(),
      )
      self._assertBeamGeneration(dl_config)


if __name__ == "__main__":
  testing.test_main()
