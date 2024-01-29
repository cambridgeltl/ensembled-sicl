import math
import torch
import random

import numpy as np

from nltk.tokenize import word_tokenize
from sentence_transformers.util import cos_sim
from datasets import concatenate_datasets

import nltk
nltk.download('punkt')


class ICRetriever(object):
    def __init__(self, icl_cfg, pool, task, seed, split):
        self.icl_cfg = icl_cfg
        if split == 'train':
            self.ic_retrieve = self.icl_cfg['retrieve'][split]
        else:
            self.ic_retrieve = self.icl_cfg['retrieve']['other']
        self.ic_retrieve_key = self.icl_cfg['retrieve_key']

        self.pool = pool

        self.task = task
        self.seed = seed

        if self.ic_retrieve in ['balance', 'label_based']:
            labels = self.pool.class_encode_column('labels').features['labels'].names
            self.balance_ic_pools = {l: self.pool.filter(lambda example: example['labels'] == l) for l in labels}
        if self.ic_retrieve in ['sbert', 'bm25', 'label_based']:
            self.build_index()

    def build_index(self):
        if self.task in ['intents', 'slots']:
            self.docs = []
            self.question = []
            for i in self.pool:
                if i['sentence'] not in self.docs:
                    self.docs.append(i['sentence'])
                if i['sent_idx'] == 0:
                    self.question.append(i['question'])
        else:
            self.docs = [i['sentence'] for i in self.pool]

        if self.ic_retrieve in ['sbert']:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer('all-mpnet-base-v2').to(
                device)  # todo: currently this is a fixed model. add the args later
            self.embedding = self.model.encode(self.docs)
        elif self.ic_retrieve in ['bm25']:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([word_tokenize(i) for i in self.docs])

    def recover_index(self, doc, question):
        assert self.task in ['intents', 'slots'], (
            ValueError('only intents and slots tasks need to recover index due to the ic example retrieval. '))
        doc_idx = self.docs.index(doc)
        question_idx = self.question.index(question)
        example_idx = question_idx + doc_idx * len(self.question)
        return example_idx

    def retrieve(self, item, current_id, split, ic_num):
        # fixme: this doesn't matter but can be removed in the final version.
        if split != 'train':
            random.seed(current_id + self.seed)
        test_input = item['sentence']
        test_question = item['question']        # only useful in intents and slots task

        if self.ic_retrieve == 'random':
            examples = self.random_retrieve(
                current_id=current_id,
                ic_num=ic_num,
                split=split
            )
        elif self.ic_retrieve == 'balance':
            examples = self.balance_retrieve(
                current_id=current_id,
                item=item,
                ic_num=ic_num,
                split=split
            )
        elif self.ic_retrieve == 'bm25':
            examples = self.bm25_retrieve(
                current_id=current_id,
                item=item,
                ic_num=ic_num,
                split=split
            )
        elif self.ic_retrieve == 'sbert':
            examples = self.sbert_retrieve(
                current_id=current_id,
                item=item,
                ic_num=ic_num,
                split=split
            )
        elif self.ic_retrieve == 'label_based':
            examples = self.label_based_retrieve(
                current_id=current_id,
                item=item,
                ic_num=ic_num,
                split=split
            )
        else:
            raise ValueError(f'Unsupported ic_retrieve method: {self.ic_retrieve}')
        examples = [i for i in examples if i['sentence'] != test_input or i['question'] != test_question][-ic_num:]
        return examples

    def random_retrieve(self, current_id, ic_num, split):
        ic_example_id_pool = list(range(len(self.pool)))
        if split == 'train':
            ic_example_id_pool.remove(current_id)
        example_ids = random.sample(ic_example_id_pool, ic_num+1)
        examples = self.pool.select(example_ids)
        return examples

    def balance_retrieve(self, current_id, item, ic_num, split):
        if ic_num+1 > len(self.balance_ic_pools):
            balance_per_ic_num = math.ceil((ic_num+1) / len(self.balance_ic_pools))
        else:
            balance_per_ic_num = 1
        examples = []
        pool_candidates = list(self.balance_ic_pools.keys())
        # pool_candidates.remove(item['labels'])
        pool_names = random.sample(pool_candidates, math.ceil((ic_num+1) / balance_per_ic_num))
        for pool_name in pool_names:
            ic_pool = self.balance_ic_pools[pool_name]
            example_ids = random.sample(range(len(ic_pool)), balance_per_ic_num)
            per_examples = ic_pool.select(example_ids)
            examples += list(per_examples)
        if len(examples) > (ic_num+1):
            random.shuffle(examples)
            examples = examples[:ic_num+1]
        return examples

    def bm25_retrieve(self, current_id, item, ic_num, split):
        test_input = item['sentence']  # todo: here should be the self.ic_retrieve_key, but we keep it 'sentence' for now
        scores = self.bm25.get_scores(word_tokenize(test_input))
        ordered_selected_idx = list(np.argsort(scores)[::-1])[:ic_num + 1]

        # if we want to go with the same question for the input in intents and slots task
        if self.task in ['intents', 'slots']:
            # default reverse order, most similar one appears at the end (most close to test input)
            docs = []
            for selected_id in ordered_selected_idx[::-1]:
                docs.append(self.docs[selected_id])
            question = item['question']
            example_ids = []
            for doc in docs:
                example_ids.append(self.recover_index(doc=doc, question=question))
            examples = self.pool.select(example_ids)
            # fixme: simply for debugging whether the recover_index works or not
            debug_docs = []
            debug_question = []
            for e in examples:
                debug_docs.append(e['sentence'])
                debug_question.append(e['question'])

        elif self.task in ['sst2', 'sst5', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos']:
            examples = self.pool.select(ordered_selected_idx[::-1])
        else:
            raise ValueError(f'Unsupported task for retriever: {self.task}. ')
        return examples

    def sbert_retrieve(self, current_id, item, ic_num, split):
        test_question = item['sentence']  # todo: here should be the self.ic_retrieve_key
        test_question_embed = self.model.encode(test_question, show_progress_bar=False)
        rel_score = cos_sim(test_question_embed, self.embedding)[0]
        ordered_selected_idx = torch.tensor(rel_score).topk(ic_num + 1)[1].cpu().tolist()

        if self.task in ['intents', 'slots']:
            docs = []
            # default reverse order, most similar one appears at the end (most close to test input)
            for selected_id in ordered_selected_idx[::-1]:
                docs.append(self.docs[selected_id])
            question = item['question']
            example_ids = []
            for doc in docs:
                example_ids.append(self.recover_index(doc=doc, question=question))
            examples = self.pool.select(example_ids)
        elif self.task in ['sst2', 'sst5', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos']:
            # default reverse order, most similar one appears at the end (most close to test input)
            examples = self.pool.select(ordered_selected_idx[::-1])
        else:
            raise ValueError(f'Unsupported task for retriever: {self.task}. ')

        return examples

    def label_based_retrieve(self, current_id, item, ic_num, split):
        if split == 'train':
            same_label_ic_num, diff_label_ic_num = (ic_num+1)//2, (ic_num+1)//2
            same_label_example = self.same_label_retrieve(current_id, item, same_label_ic_num, split)
            diff_label_example = self.different_label_retrieve(current_id, item, diff_label_ic_num, split)
            return same_label_example + diff_label_example
        else:
            return self.random_retrieve(current_id, ic_num, split)

    def same_label_retrieve(self, current_id, item, ic_num, split):
        # only used when training because of the label being used
        assert split == 'train'

        label = item['labels']
        pool = self.balance_ic_pools[label]
        pool_size = len(pool)
        if ic_num > pool_size:
            ic_num = pool_size
        example_ids = random.sample(range(len(pool)), ic_num)
        examples = pool.select(example_ids)
        return list(examples)

    def different_label_retrieve(self, current_id, item, ic_num, split):
        # only used when training because of the label being used
        assert split == 'train'

        label = item['labels']
        example_ids = []
        if self.task in ['intents', 'slots']:
            doc = item['sentence']
            for q in self.question:
                if q != item['question']:
                    example_ids.append(self.recover_index(doc=doc, question=q))
            example_pool_size = len(example_ids)
            if ic_num + 1 > example_pool_size:
                ic_num = example_pool_size
            random.shuffle(example_ids)
            examples = self.pool.select(example_ids[:ic_num])
        elif self.task in ['sst5', 'sst2', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos']:
            # label_choices = [k for k in self.balance_ic_pools.keys() if k != label]
            # examples = []
            # pool_size = sum([len(v) for k, v in self.balance_ic_pools.items() if k != label])
            # if ic_num + 1 > pool_size:
            #     ic_num = pool_size
            # for i in range(ic_num*10):
            #     label_type = random.choice(label_choices)
            #     pool = self.balance_ic_pools[label_type]
            #     example = pool[random.randint(0, len(pool)-1)]
            #     if example not in examples:
            #         examples.append(example)
            #     if len(examples) > ic_num:
            #         break
            pool = concatenate_datasets([d for k, d in self.balance_ic_pools.items() if k != label])
            pool_size = len(pool)
            if ic_num > pool_size:
                ic_num = pool_size
            example_ids = random.sample(range(len(pool)), ic_num)
            examples = pool.select(example_ids)
        else:
            raise ValueError(f'Unsupported task {self.task}')
        return list(examples)