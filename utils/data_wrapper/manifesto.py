# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Multi3NLU dataset"""

import json
import os
import ast

import datasets

import pandas as pd
from utils.verbalizer import MANIFESTOS_MAPPING


_CITATION = """
manifesto citation
"""

_DESCRIPTION = """
The Manifesto Project analyses parties’ election manifestos in order to study parties’ policy preferences.
"""

_HOMEPAGE = "https://manifesto-project.wzb.eu/"

_LICENSE = "CC BY 4.0"

_SPLITS = ["train", "dev", "test"]

_DATA_DIR = r"./txt_data/manifestos"

_URLS = {
    "data_dir": _DATA_DIR,
    "train50_split": os.path.join(_DATA_DIR, 'train_balance.json'),
    "train50_imbalance_split": os.path.join(_DATA_DIR, 'train_imbalance.json'),
    "train500_split": os.path.join(_DATA_DIR, 'train_balance_500.json'),
    "train500_imbalance_split": os.path.join(_DATA_DIR, 'train_imbalance_500.json'),
    "dev_split": os.path.join(_DATA_DIR, 'test_balance.json'),       # todo: change the dev split to other data in the future
    "test_split": os.path.join(_DATA_DIR, 'test_balance.json'),
    "dev_imbalance_split": os.path.join(_DATA_DIR, 'test_imbalance.json'),
    "test_imbalance_split": os.path.join(_DATA_DIR, 'test_imbalance.json'),
}


class ManifestoConfig(datasets.BuilderConfig):
    """BuilderConfig for ExplainLikeImFive."""

    def __init__(self, train_imbalance, test_imbalance, train_size, **kwargs):
        """BuilderConfig for Multi3NLU.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ManifestoConfig, self).__init__(**kwargs)
        self.train_imbalance = train_imbalance
        self.test_imbalance = test_imbalance
        self.train_size = train_size


class Manifestos(datasets.GeneratorBasedBuilder):
    """MUlti3NLU dataset."""

    BUILDER_CONFIG_CLASS = ManifestoConfig
    BUILDER_CONFIGS = [
        ManifestoConfig(
            name="manifesto",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            train_imbalance=False,
            test_imbalance=False,
            train_size=50
        )
    ]

    DEFAULT_CONFIG_NAME = "manifesto"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                # 'sent_idx': datasets.Value('int32'),
                "sentence": datasets.Value('string'),
                "question": datasets.Value('string'),
                "labels": datasets.Value('string')
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = _URLS
        train_imbalance = self.config.train_imbalance
        test_imbalance = self.config.test_imbalance
        train_size = self.config.train_size         # should be [50, 100, 200, ..., all]
        train_split_name = f"train{train_size}_split" if not train_imbalance else f"train{train_size}_imbalance_split"
        dev_split_name = f"dev_split" if not test_imbalance else f"dev_imbalance_split"
        test_split_name = f"test_split" if not test_imbalance else f"test_imbalance_split"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files[train_split_name]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files[dev_split_name]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files[test_split_name]),
                },
            ),
        ]

    def _generate_examples(self, split_dir: str):
        with open(split_dir) as json_file:
            data = json.load(json_file)

        question = "Which category about US society does the sentence belong to?"

        for idx, data_item in enumerate(data):
            sent = data_item['sentence']
            label = MANIFESTOS_MAPPING[data_item['label']].lower()
            # label = data_item['label'].lower()

            yield idx, {
                        "idx": idx,
                        "sentence": sent,
                        "labels": label,
                        "question": question
            }