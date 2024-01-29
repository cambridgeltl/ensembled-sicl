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


_CITATION = """
@article{moghe2022multi3nlu++,
  title={Multi3nlu++: A multilingual, multi-intent, multi-domain dataset for natural language understanding in task-oriented dialogue},
  author={Moghe, Nikita and Razumovskaia, Evgeniia and Guillou, Liane and Vuli{\'c}, Ivan and Korhonen, Anna and Birch, Alexandra},
  journal={arXiv preprint arXiv:2212.10455},
  year={2022}
}
"""

_DESCRIPTION = """
Multi3NLU++ consists of 3080 utterances per language representing challenges in building multilingual multi-intent 
multi-domain task-oriented dialogue systems. The domains include banking and hotels. There are 62 unique intents.
"""

_HOMEPAGE = "https://huggingface.co/datasets/uoe-nlp/multi3-nlu"

_LICENSE = "CC BY 4.0"

_SPLITS = ["train", "dev", "test"]

_DATA_DIR = r"./csv_data/nlupp-splits"

_URLS = {
    "data_dir": _DATA_DIR,
    "templates": os.path.join(_DATA_DIR, 'templates.json'),
    "ontology": os.path.join(_DATA_DIR, 'ontology.json'),
    "train50_split": os.path.join(_DATA_DIR, 'train_50.csv'),
    "train100_split": os.path.join(_DATA_DIR, 'train_100.csv'),
    "train200_split": os.path.join(_DATA_DIR, 'train_200.csv'),
    "train300_split": os.path.join(_DATA_DIR, 'train_300.csv'),
    "train500_split": os.path.join(_DATA_DIR, 'train_500.csv'),
    "train800_split": os.path.join(_DATA_DIR, 'train_800.csv'),
    "train1000_split": os.path.join(_DATA_DIR, 'train_1000.csv'),
    "train1500_split": os.path.join(_DATA_DIR, 'train_1500.csv'),
    "train2000_split": os.path.join(_DATA_DIR, 'train_2000.csv'),
    "trainall_split": os.path.join(_DATA_DIR, 'train_all.csv'),
    "full": os.path.join(_DATA_DIR, 'full_nlupp.csv'),
    "dev_split": os.path.join(_DATA_DIR, 'test.csv'),       # todo: change the dev split to other data in the future
    "test_split": os.path.join(_DATA_DIR, 'test.csv'),
}


class Multi3NLUConfig(datasets.BuilderConfig):
    """BuilderConfig for ExplainLikeImFive."""

    def __init__(self, train_size, nlu_task, **kwargs):
        """BuilderConfig for Multi3NLU.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Multi3NLUConfig, self).__init__(**kwargs)
        self.train_size = train_size
        self.nlu_task = nlu_task


class Multi3NLU(datasets.GeneratorBasedBuilder):
    """MUlti3NLU dataset."""

    BUILDER_CONFIG_CLASS = Multi3NLUConfig
    BUILDER_CONFIGS = [
        Multi3NLUConfig(
            name="multi3nlu",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            train_size=50,
            nlu_task='intents'
        )
    ]

    DEFAULT_CONFIG_NAME = "multi3nlu"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                'sent_idx': datasets.Value('int32'),
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
        train_size = self.config.train_size         # should be [50, 100, 200, ..., all]
        task = self.config.nlu_task                     # should be [intents, slots]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files[f"train{train_size}_split"]),
                    'task': task
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files["dev_split"]),
                    "task": task
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "split_dir": os.path.join(downloaded_files["test_split"]),
                    "task": task
                },
            ),
        ]

    def _generate_examples(self, split_dir: str, task: str):
        with open(_URLS['ontology']) as json_file:
            ontology = json.load(json_file)

        # todo: the domain in the data?
        if task == "slots":
            # intent_desc_dict = {key: ontology["slots"][key]["description"] for key in ontology["slots"].keys() if
            #                     "general" in ontology["slots"][key]["domain"] or domain in ontology["slots"][key][
            #                         "domain"]}
            intent_desc_dict = {key: ontology["slots"][key]["description"] for key in ontology["slots"].keys()}
        elif task == 'intents':
            # intent_desc_dict = {key: ontology["intents"][key]["description"][14:-1] for key in
            #                     ontology["intents"].keys() if
            #                     "general" in ontology["intents"][key]["domain"] or domain in
            #                     ontology["intents"][key]["domain"]}
            intent_desc_dict = {key: ontology["intents"][key]["description"][14:-1] for key in
                                ontology["intents"].keys()}
            for intent, description in intent_desc_dict.items():
                if not description.startswith("to "):
                    intent_desc_dict[intent] = description.replace("asking", "to ask")

        intents_or_slots_list = sorted(list(intent_desc_dict.keys()))

        data = pd.read_csv(split_dir).T.to_dict()

        idx = 0
        for sent_idx, example in enumerate(data.values()):
            if task == 'slots':
                data_item = make_template_slot(
                    idx=idx,
                    example=example,
                    intent_desc_dict=intent_desc_dict,
                    intents_or_slots_list=intents_or_slots_list
                )
                if data_item is not None:
                    if len(data_item) > 0:
                        for item in data_item:
                            yield idx, {
                                "idx": idx,
                                "sent_idx": sent_idx,
                                **item
                            }
                            idx += 1
            elif task == 'intents':
                data_item = make_template(
                    idx=idx,
                    example=example,
                    intent_desc_dict=intent_desc_dict,
                    intents_or_slots_list=intents_or_slots_list
                )
                if data_item is not None:
                    if len(data_item) > 0:
                        for item in data_item:
                            yield idx, {
                                "idx": idx,
                                "sent_idx": sent_idx,
                                **item
                            }
                            idx += 1


def make_template_slot(idx, example, intent_desc_dict, intents_or_slots_list):
    sentence = example["text"]
    if str(sentence) == "nan":
        return None

    slots = example.get("slots")

    if str(slots) == 'nan':
        slots = {}
    else:
        slots = ast.literal_eval(slots)
        slots = {slot: slots[slot]['value'] for slot in slots.keys()}

    data = []
    for slot in intents_or_slots_list:
        data_item = dict()
        data_item['sentence'] = sentence
        data_item['question'] = intent_desc_dict[slot]
        if slot in list(slots.keys()):
            data_item['labels'] = slots[slot]
        else:
            data_item['labels'] = "unanswerable"
        data.append(data_item)
    return data


def make_template(idx, example, intent_desc_dict, intents_or_slots_list):
    yes, no = "yes", "no"

    sentence = example["text"]
    if str(sentence)=="nan":
        return None
    intents = example.get("intents")

    if str(intents) == 'nan':
        intents = []
    else:
        intents = ast.literal_eval(intents)

    data = []
    for intent in intents_or_slots_list:

        data_item = dict()
        data_item['sentence'] = sentence
        data_item['question'] = "did the user intend " + intent_desc_dict[intent]
        if intent in intents:
            data_item["labels"] = yes
        else:
            data_item["labels"] = no
        data.append(data_item)
    return data