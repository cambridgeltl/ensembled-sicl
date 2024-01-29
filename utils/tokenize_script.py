import os
import math
import torch
import json
import random
import copy

from torch.utils.data import Dataset

from utils.retriever import ICRetriever


# TODO: icl cfg: number of ic examples, order, ...,
#  and the prompt template, prepare the ensemble with identical id
class TokenizedDataset(Dataset):
    def __init__(self,
                 dataset,
                 processor,
                 split,
                 input_max_length,
                 label_max_length,
                 mode,
                 input_format,
                 **kwargs):
        self.processor = processor
        self.split = split
        self.dataset = dataset

        self.ic_examples_record = [''] * len(self.dataset)
        self.input_record = [''] * len(self.dataset)

        self.mode = mode
        self.model_type = kwargs['model_type']  # 'google/flan-t5', 'meta-llama/Llama-2-7b-hf'
        # if self.model_type.startswith('google/flan-t5'):
        self.input_max_length = input_max_length
        self.label_max_length = label_max_length
        # else:
        #     self.input_max_length = input_max_length + label_max_length
        #     self.label_max_length = input_max_length + label_max_length

        if kwargs['icl_cfg'] is not None:
            self.use_ic = True
            self.icl_cfg = kwargs['icl_cfg']
            self.ic_num = self.icl_cfg['ic_num']
            self.ic_order = self.icl_cfg['order']
            self.ic_pool = kwargs['ic_pools'][self.icl_cfg['ic_pool']]
            self.ic_retrieve = self.icl_cfg['retrieve']
            self.ic_retrieve_key = self.icl_cfg['retrieve_key']

            self.retriever = ICRetriever(
                icl_cfg=self.icl_cfg,
                pool=kwargs['ic_pools'][self.icl_cfg['ic_pool']],
                task=kwargs['task'],
                seed=kwargs['seed'],
                split=self.split
            )
            if self.split != 'train':
                self.non_train_ic_examples = [[]] * len(self.dataset)
        else:
            self.use_ic = False

        if input_format is None:
            self.template = []
            self.ic_template = []
            self.input_template = []
            prompt_dir = os.path.join('prompt', kwargs['task'])
            prompt_files = os.listdir(prompt_dir)
            for p in prompt_files:
                format_json = os.path.join(prompt_dir, p)
                with open(format_json) as f:
                    template = json.load(f)

                self.template.append(template['template'])
                self.ic_template.append(template.get('example_template', '<INPUT>\n<LABEL>'))
                self.input_template.append(template.get('input_template', '<SENTENCE>\n<QUESTION>'))
            self.template_num = len(self.template)
        else:
            format_json = os.path.join('prompt', kwargs['task'], input_format[0] + '.json')
            with open(format_json) as f:
                template = json.load(f)

            self.template = [template['template']]
            self.ic_template = [template.get('example_template', '<INPUT>\n<LABEL>')]
            self.input_template = [template.get('input_template', '<SENTENCE>\n<QUESTION>')]
            self.template_num = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        template_id = random.choice(list(range(self.template_num)))
        if self.use_ic:
            examples = None
            use_existed_examples = False
            if self.split != 'train':
                if len(self.non_train_ic_examples[idx]) != 0:
                    examples = self.non_train_ic_examples[idx]
                    use_existed_examples = True

            if examples is None:
                examples = self.retriever.retrieve(
                    item=item,
                    current_id=idx,
                    split=self.split,
                    ic_num=self.ic_num
                )

            if self.ic_order == 'random' and not use_existed_examples:
                random.shuffle(examples)

            if self.split != 'train':
                self.non_train_ic_examples[idx] = examples

            ic_strs = []
            for example in examples:
                example_input = (self.input_template[template_id].replace('<SENTENCE>', example['sentence'].strip())
                                 .replace('<QUESTION>', example['question'].strip()))
                example_str = (self.ic_template[template_id].replace('<INPUT>', example_input.strip())
                               .replace('<LABEL>', example['labels'].strip()))
                ic_strs.append(example_str)
            ic_examples = '\n\n'.join(ic_strs)
        else:
            ic_examples = ''

        item_input = (self.input_template[template_id].replace('<SENTENCE>', item['sentence'].strip())
                                 .replace('<QUESTION>', item['question'].strip()))
        input_str = (self.template[template_id].replace('<EXAMPLE_PAIRS>', ic_examples.strip())
                     .replace('<INPUT>', item_input.strip()))
        if (not self.use_ic and
                len(self.template[template_id].replace('<EXAMPLE_PAIRS>', '').replace('<INPUT>', '').strip()) == 0):
            input_str = input_str.strip() + '\n'
        label_str = item['labels']
        self.input_record[idx] = input_str

        if self.model_type.startswith('google/flan-t5') or self.model_type.startswith('t5'):
            tokenized_input = self.processor(
                input_str,
                padding='max_length',
                truncation=True,
                max_length=self.input_max_length          # todo: modify the length
            )
            tokenized_inferred = self.processor(
                label_str,
                padding='max_length',
                truncation=True,
                max_length=self.label_max_length          # todo: modify the length
            )
            tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
            # Here -100 will let the model not to compute the loss of the padding tokens.
            tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.processor.pad_token_id] = -100
            item = {
                'input_ids': torch.LongTensor(tokenized_input.data["input_ids"]),
                'attention_mask': torch.LongTensor(tokenized_input.data["attention_mask"]),
                'labels': tokenized_inferred_input_ids,
            }
        else:
            # todo: adds for llama
            if self.split == 'train':
                tokenized_input_label = self.processor(
                    input_str + " " + label_str,
                    padding='max_length',
                    max_length=self.input_max_length + self.label_max_length
                )
                tokenized_inferred = self.processor(
                    label_str, add_special_tokens=False
                )

                input_ids = tokenized_input_label['input_ids']              # input and labels, left padded
                label_ids = ([-100] * (
                        self.input_max_length + self.label_max_length - len(tokenized_inferred['input_ids']))
                             + tokenized_inferred['input_ids'])             # only labels and left pad with -100 to avoid loss
                # label_ids = tokenized_input_label['input_ids']
                # label_ids[label_ids == self.processor.pad_token_id] = -100
                attention_mask = tokenized_input_label['attention_mask']
                assert len(input_ids) == len(label_ids)
                item = {
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'labels': torch.LongTensor(label_ids),
                }
            else:
                # when inference, it becomes more complicated:
                #  1) the input should not conclude the label when using model.generate
                #  2) in order to compute the prediction loss, we have to have the whole sequence (input + label)
                #  and the label when feeding the input to model(**inputs)
                tokenized_input = self.processor(
                    input_str,
                    padding='max_length',
                    max_length=self.input_max_length
                )
                tokenized_inferred = self.processor(
                    label_str, add_special_tokens=False
                )
                input_ids = tokenized_input['input_ids']
                attention_mask = tokenized_input['attention_mask']
                label_ids = [-100] * (
                            self.input_max_length + self.label_max_length - len(tokenized_inferred['input_ids'])) + \
                            tokenized_inferred['input_ids']

                tokenized_input_label = self.processor(
                    input_str + label_str,
                    padding='max_length',
                    max_length=self.input_max_length + self.label_max_length
                )
                total_seq = tokenized_input_label['input_ids']
                total_seq_attention_mask = tokenized_input_label['attention_mask']
                assert len(total_seq) == len(label_ids)

                item = {
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'labels': torch.LongTensor(label_ids),
                    'total_seq': torch.LongTensor(total_seq),
                    'total_seq_attention_mask': torch.LongTensor(total_seq_attention_mask),
                }
        return item