import os
import random
import json

from datasets import load_dataset, Features, Value, concatenate_datasets
from utils.tokenize_script import TokenizedDataset
from utils.verbalizer import SST5_NEW_LABEL_MAP, RTE_ID2VERB, ANLI_ID2VERB

DATA_DIR = 'data'
PROCESSED_DATA_DIR = 'preprocessed_data'


def load_data(dataset_name, data_args, imbalance, test_imbalance):
    # todo: modify the loading scripts below to fit in the dataset load.py in data/load_scripts
    if dataset_name in ['multi3nlu']:
        data = load_dataset('utils/data_wrapper/multi3nlu.py',
                            train_size=data_args['train_size'],
                            nlu_task=data_args['task'])
    elif dataset_name in ['manifestos']:
        data = load_dataset('utils/data_wrapper/manifesto.py',
                            train_size=data_args['train_size'],
                            train_imbalance=imbalance,
                            test_imbalance=test_imbalance)
    elif dataset_name in ['hate_speech']:
        data = load_dataset("ucberkeley-dlab/measuring-hate-speech", 'binary')
        train_size = data_args['train_size']
        sample_indices = list(range(int(train_size)))

        all_columns = data.column_names['train']
        all_columns.remove('hate_speech_score')
        all_columns.remove('text')

        question = "Is the sentence hate, neutral or support? "

        data['train'] = data['train'].remove_columns(all_columns).add_column("question", [question] * len(data['train']))
        train_data = data['train'].select(sample_indices).map(map_hate_speech).remove_columns(['hate_speech_score']).rename_column('text', 'sentence')
        dev_data = data['train'].select(list(range(2000, 2500))).map(map_hate_speech).remove_columns(['hate_speech_score']).rename_column('text', 'sentence')
        test_data = data['train'].select(list(range(2500, 4000))).map(map_hate_speech).remove_columns(['hate_speech_score']).rename_column('text', 'sentence')
        data['train'] = train_data
        data['dev'] = dev_data
        data['test'] = test_data

    elif dataset_name in ['sst2']:
        data = load_dataset("SetFit/sst2")

        if imbalance:
            labels = data['train'].class_encode_column('label_text').features['label_text'].names
            most_imbalance_portion = 0.9
            most_imbalance_num = int(int(data_args['train_size']) * most_imbalance_portion)
            most_imbalance_label = "negative"
            other_num = int((int(data_args['train_size']) - most_imbalance_num) / (len(labels) - 1))+1
            data_list = []
            for label in labels:
                if label == most_imbalance_label:
                    num = most_imbalance_num
                else:
                    num = other_num
                data_list.append(
                    data['train'].filter(lambda example: example['label_text'] == label).select(list(range(num))))
            data['train'] = concatenate_datasets(data_list)

        train_size = data_args['train_size']
        sample_indices = list(range(int(train_size)))
        data['train'] = data['train'].select(sample_indices)
        question = "is this sentence 'positive' or 'negative'? "
        for k in data.keys():
            data[k] = (data[k].rename_column('text', 'sentence')
                       .rename_column('label_text', 'labels')
                       .remove_columns(['label'])
                       .add_column("question", [question] * len(data[k])))
    elif dataset_name in ['sst5']:
        data = load_dataset("SetFit/sst5")

        if imbalance:
            labels = data['train'].class_encode_column('label_text').features['label_text'].names
            most_imbalance_portion = 0.8
            most_imbalance_num = int(int(data_args['train_size']) * most_imbalance_portion)
            most_imbalance_label = "negative"
            other_num = int((int(data_args['train_size']) - most_imbalance_num) / (len(labels) - 1))+1
            data_list = []
            for label in labels:
                if label == most_imbalance_label:
                    num = most_imbalance_num
                else:
                    num = other_num
                data_list.append(
                    data['train'].filter(lambda example: example['label_text'] == label).select(list(range(num))))
            data['train'] = concatenate_datasets(data_list)

        if test_imbalance:
            for k in data.keys():
                if k != 'train':
                    labels = data[k].class_encode_column('label_text').features['label_text'].names
                    most_imbalance_portion = 0.8
                    most_imbalance_label = "negative"
                    most_imbalance_num = len(data[k].filter(lambda example: example['label_text'] == most_imbalance_label))
                    rest_num = int(most_imbalance_num / most_imbalance_portion) - most_imbalance_num
                    other_num = int(rest_num / (len(labels) - 1) + 1)
                    data_list = []
                    for label in labels:
                        if label == most_imbalance_label:
                            data_list.append(data[k].filter(lambda example: example['label_text'] == label))
                        else:
                            data_list.append(
                                data[k].filter(lambda example: example['label_text'] == label).select(list(range(other_num))))
                    data[k] = concatenate_datasets(data_list)

        train_size = data_args['train_size']
        sample_indices = list(range(int(train_size)))
        data['train'] = data['train'].select(sample_indices)
        question = "is this sentence 'great', 'good', 'neutral', 'bad' or 'terrible'? "
        for k in data.keys():
            data[k] = (data[k].rename_column('text', 'sentence')
                       .rename_column('label_text','labels')
                       .remove_columns(['label'])
                       .add_column("question", [question]*len(data[k])))
            data[k] = data[k].map(map_sst5_label, load_from_cache_file=False)
    elif dataset_name in ['rte']:
        data = load_dataset("glue", "rte")
        train_size = data_args['train_size']
        sample_indices = list(range(int(train_size)))
        data['train'] = data['train'].select(sample_indices)
        question = "Does Sentence1 entails Sentence2?"
        for k in data.keys():
            if k == 'test':
                data[k] = data['validation']
            else:
                data[k] = data[k].add_column('labels', [""]*len(data[k]))
                data[k] = data[k].map(map_rte_sentences, load_from_cache_file=False)
                data[k] = (data[k].remove_columns(['sentence1', 'sentence2', 'label'])
                           .add_column("question", [question]*len(data[k])))
    elif dataset_name in ['anli']:
        data = load_dataset('anli')

        if imbalance:
            labels = data['train_r1'].features['label'].names
            most_imbalance_portion = 0.8
            most_imbalance_num = int(int(data_args['train_size']) * most_imbalance_portion)
            most_imbalance_label = "maybe"
            other_num = int((int(data_args['train_size']) - most_imbalance_num) / (len(labels) - 1))+1
            data_list = []
            for label in labels:
                if label == most_imbalance_label:
                    num = most_imbalance_num
                else:
                    num = other_num
                data_list.append(
                    data['train_r1'].filter(lambda example: example['label'] == label).select(list(range(num))))
            data['train_r1'] = concatenate_datasets(data_list)

        train_size = data_args['train_size']
        sample_indices = list(range(0, 0+int(train_size)))
        ks = ['train', 'dev', 'test']
        question = "Does Sentence1 entails Sentence2?"
        for k in ks:
            data[k] = data[k+"_r1"]
            if k == 'train':
                data[k] = data[k].select(sample_indices)
            data[k] = data[k].add_column('labels', [""] * len(data[k]))
            data[k] = data[k].map(map_anli_sentences, load_from_cache_file=False)
            data[k] = (data[k].remove_columns(['uid', 'premise', 'label', 'reason'])
                       .add_column("question", [question] * len(data[k])))
    elif dataset_name in ['causal_judgment', 'cause_and_effect']:
        data = load_dataset('tasksource/bigbench', dataset_name)
        train_size = data_args['train_size']
        assert int(train_size) < 100, ValueError(f'train size for {dataset_name} should be lower than 100')
        sample_indices = list(range(int(train_size)))
        ks = ['train', 'validation', 'test']
        if dataset_name == 'causal_judgment':
            question = "How would a typical person answer each of the questions about causation?"
        else:
            question = "Does Sentence1 make more sense than Sentence2?"
            index_list = list(range(0, 41)) + list(range(82, 123))
            random.shuffle(index_list)
            data['train'] = data['train'].select(index_list)
        for k in ks:
            if k in ['train', 'validation']:
                if k == 'train':
                    data[k] = data[k].select(sample_indices)
                else:
                    if dataset_name == 'cause_and_effect':
                        data[k] = data[k].select(list(range(0, 10)) + list(range(20, 30)))
                data[k] = data[k].add_column('labels', [""] * len(data[k]))
                map_function = map_causal_judgment_sentences if dataset_name == 'causal_judgment' else map_cause_and_effect_sentences
                if k != 'train' and dataset_name == 'cause_and_effect' and test_imbalance:
                    map_function = imb_map_cause_and_effect_sentences
                if k == 'train' and dataset_name == 'cause_and_effect' and imbalance:
                    map_function = imb_map_cause_and_effect_sentences
                data[k] = data[k].map(map_function, load_from_cache_file=False)
                data[k] = (data[k].remove_columns(['inputs', 'targets', 'multiple_choice_targets', 'multiple_choice_scores'])
                           .add_column("question", [question] * len(data[k])))
            else:
                data[k] = data['validation']
    else:
        raise ValueError(f'{dataset_name} not supported! add the loading scripts in utils/load_data.py')
    return data


def tokenize_dataset(train_split, eval_split, test_split, processor, **kwargs):
    tokenized_data = dict()
    if kwargs['input_format'] is None:
        prompt_dir = os.path.join('prompt', kwargs['task'])
        prompt_files = os.listdir(prompt_dir)
        max_source_lens = []
        for p in prompt_files:
            with open(os.path.join(prompt_dir, p)) as f:
                template = json.load(f)
            template_str = ' '.join(list(template.values())).replace('<EXAMPLE_PAIRS>', '').replace('<INPUT>',
                                                                                                    '').replace(
                '<LABEL>', '').replace('<SENTENCE>', '').replace('<QUESTION>', '')
            tokenized_inputs = concatenate_datasets([train_split, test_split]).map(
                lambda x: processor(x["sentence"] + x['question'] + x['labels'], truncation=True), batched=True,
                remove_columns=train_split.column_names)
            max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
            max_source_length = max_source_length + len(
                processor(template_str).data["input_ids"]) + 5  # 5 is for just in case
            max_source_lens.append(max_source_length)
        max_source_length = max(max_source_lens)
    else:
        with open(os.path.join('prompt', kwargs['task'], kwargs['input_format'][0] + '.json')) as f:
            template = json.load(f)
        template_str = ' '.join(list(template.values())).replace('<EXAMPLE_PAIRS>', '').replace('<INPUT>', '').replace(
            '<LABEL>', '').replace('<SENTENCE>', '').replace('<QUESTION>', '')
        tokenized_inputs = concatenate_datasets([train_split, test_split]).map(
            lambda x: processor(x["sentence"] + x['question'] + x['labels'], truncation=True), batched=True,
            remove_columns=train_split.column_names)
        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        max_source_length = max_source_length + len(
            processor(template_str).data["input_ids"]) + 5  # 5 is for just in case

    if kwargs['icl_cfg'] is not None:
        max_source_length = max_source_length * (kwargs['icl_cfg']['ic_num'] + 1)
    print(f"Max source length: {max_source_length}")
    tokenized_targets = concatenate_datasets([train_split, test_split]).map(
        lambda x: processor(x["labels"], truncation=True), batched=True,
        remove_columns=train_split.column_names)
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]]) + 5
    print(f"Max target length: {max_target_length}")

    if train_split:
        tokenized_train = TokenizedDataset(
            dataset=train_split,
            split='train',
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
            # todo: prompt template
        )
        tokenized_data['train'] = tokenized_train
    if eval_split:
        tokenized_eval = TokenizedDataset(
            dataset=eval_split,
            split='eval',
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['eval'] = tokenized_eval
    if test_split:
        tokenized_test = TokenizedDataset(
            dataset=test_split,
            split='test',
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['test'] = tokenized_test
    return tokenized_data, max_source_length, max_target_length


def map_sst5_label(example):
    example['labels'] = SST5_NEW_LABEL_MAP[example['labels']]
    return example

def map_hate_speech(example):
    if float(example['hate_speech_score']) > 0.5:
        label = 2
    elif float(example['hate_speech_score']) < -1:
        label = 0
    else:
        label = 1
    HATE_SPEECH_LABEL = {
        0: 'support',
        1: 'neutral',
        2: 'hate'
    }
    example['labels'] = HATE_SPEECH_LABEL[label]
    return example


def map_rte_sentences(example):
    example['sentence'] = "Sentence1: " + example['sentence1'] + "\nSentence2: " + example['sentence2']
    # flan provides one template: Does <sentence_1> means that <sentence_2>?
    example['labels'] = RTE_ID2VERB[example['label']]
    return example


def map_anli_sentences(example):
    example['sentence'] = "Premise: " + example['premise'] + "\nHypothesis: " + example['hypothesis']
    # flan provides one template: Does <sentence_1> means that <sentence_2>?
    example['labels'] = ANLI_ID2VERB[example['label']]
    return example


def map_causal_judgment_sentences(example):
    example['sentence'] = example['inputs'].split("\n\n\n")[-1].strip("QA:")
    example['labels'] = "yes" if 'Yes' in example['targets'] else "no"
    return example


def map_cause_and_effect_sentences(example):
    input_str = example['inputs']
    choices = input_str.split('answer:')[0].split('choice:')[1:]
    choices = [i.strip() for i in choices]
    if len(choices) == 0:
        choices = example['multiple_choice_targets']
    # choices = example['multiple_choice_targets']
    targets = example['targets'][0].strip()
    if example['inputs'].startswith('For each example, two events are given.'):
        question = "Two sentences are given. Does Sentence1 cause Sentence2?"
        example['sentence'] = question + "\n" + f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    elif len(example['inputs']) == 0:
        example['sentence'] = f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    else:
        question = "Does Sentence1 make more sense than Sentence2?"
        example['sentence'] = question + "\n" + f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    example['labels'] = "yes" if choices.index(targets) == 0 else "no"
    return example


def imb_map_cause_and_effect_sentences(example):
    # If has [::-1] then the imbalance (more) label is no, else is yes
    choices = example['multiple_choice_targets'][::-1]
    targets = example['targets'][0].strip()
    if example['inputs'].startswith('For each example, two events are given.'):
        question = "Two sentences are given. Does Sentence1 cause Sentence2?"
        example['sentence'] = question + "\n" + f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    elif len(example['inputs']) == 0:
        example['sentence'] = f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    else:
        question = "Does Sentence1 make more sense than Sentence2?"
        example['sentence'] = question + "\n" + f"Sentence1: {choices[0]}\nSentence2: {choices[1]}"
    example['labels'] = "yes" if choices.index(targets) == 0 else "no"
    return example