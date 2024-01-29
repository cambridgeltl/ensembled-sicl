import os
import json
import argparse
import yaml
import ast
import shutil

from tqdm import tqdm
from utils.evaluator import EvaluateTool
from utils.verbalizer import VERBALIZER
from utils.run_config import create_run_name

import numpy as np


def load_result_json(folder_dir):
    if 'predictions_predict.json' in os.listdir(folder_dir):
        try:
            with open(os.path.join(folder_dir, 'predictions_predict.json')) as f:
                prediction = json.load(f)
        except:
            print(folder_dir)
        return prediction
    else:
        raise FileNotFoundError(f"{folder_dir} doesn't contain predictions_predict.json")


def check_results_num(preds):
    nums = []
    for v in preds.values():
        nums.append(len(v))
    if len(set(nums)) == 1:
        return True
    else:
        return False


def assert_equal(p_str, g_str, mode='loose'):
    if p_str == g_str:
        return True

    if mode == 'loose':
        if p_str in g_str or g_str in p_str:
            return True

    return False



class Voters():
    def __init__(self, task, pools, strategy, keep_distinct=False, min_votes=1, with_logprobs=False):
        self.voter_num = len(pools)
        self.voter_names = []
        self.voter_pools = []
        self.strategy = strategy

        for k, v in pools.items():
            self.voter_names.append(k)
            self.voter_pools.append(v)

        self.keep_distinct = keep_distinct
        assert min_votes <= self.voter_num, ValueError(f'min_votes should smaller than voter_num {self.voter_num}. ')
        self.min_votes = min_votes
        self.with_logprobs = with_logprobs
        self.task = task

    def vote(self):
        if self.strategy == 'majority_vote':
            return self.majority_vote()
        if self.strategy == 'mean_prob':
            return self.mean_logprobs()
        if self.strategy == 'max_prob':
            return self.max_logprobs()
        else:
            raise NotImplementedError

    def mean_logprobs(self):
        postprocessed_preds = []
        postprocessed_logprobs = []

        for id in tqdm(range(len(self.voter_pools[0]))):
            logprobs = [ast.literal_eval(t[id]['logprob']) for t in self.voter_pools]
            mean_logprobs = np.mean(np.array(logprobs), axis=0)
            mean_label = np.argmax(mean_logprobs)
            postprocessed_preds.append(VERBALIZER[self.task][mean_label])
            postprocessed_logprobs.append((mean_logprobs / sum(mean_logprobs)).tolist())
        return postprocessed_preds, postprocessed_logprobs

    def max_logprobs(self):
        postprocessed_preds = []
        postprocessed_logprobs = []

        for id in tqdm(range(len(self.voter_pools[0]))):
            logprobs = [ast.literal_eval(t[id]['logprob']) for t in self.voter_pools]
            max_logprobs = np.max(np.array(logprobs), axis=0)
            max_label = np.argmax(max_logprobs)
            postprocessed_preds.append(VERBALIZER[self.task][max_label])
            postprocessed_logprobs.append((max_logprobs / sum(max_logprobs)).tolist())
        return postprocessed_preds, postprocessed_logprobs

    def majority_vote(self):
        postprocessed_preds = []
        if self.with_logprobs:
            postprocessed_logprobs = []
        else:
            postprocessed_logprobs = None

        for id in tqdm(range(len(self.voter_pools[0]))):
            items = [t[id]['prediction'] for t in self.voter_pools]
            logprobs = [ast.literal_eval(t[id]['logprob']) for t in self.voter_pools]

            if not self.with_logprobs:
                eq_matrix = np.zeros((self.voter_num, self.voter_num), dtype=bool)
                for i in range(self.voter_num):
                    for j in range(i + 1, self.voter_num):
                        eq_matrix[i][j] = assert_equal(p_str=items[i], g_str=items[j], mode='loose')
                eq_matrix = eq_matrix + eq_matrix.T + np.identity(self.voter_num, dtype=bool)

            else:
                eq_matrix = np.zeros((self.voter_num, self.voter_num), dtype=float)
                for i in range(self.voter_num):
                    for j in range(self.voter_num):
                        if assert_equal(p_str=items[i], g_str=items[j], mode='loose'):
                            eq_matrix[i][j] = max(logprobs[j])

            same_votes_num = eq_matrix.sum(-1)

            max_vote = same_votes_num.max()
            if max_vote >= self.min_votes:
                # at least min_votes voters have same results
                keep_indices = (same_votes_num == max_vote).nonzero()[0]
            else:
                keep_indices = list(range(self.voter_num))

            if self.keep_distinct:
                keep_results = items[keep_indices[0]]
                if self.with_logprobs:
                    keep_logprobs = np.mean([logprobs[k_i] for k_i in keep_indices], axis=0).tolist()
            else:
                keep_results = [items[k_i] for k_i in keep_indices]
                if self.with_logprobs:
                    keep_logprobs = [logprobs[k_i] for k_i in keep_indices]

            postprocessed_preds.append(keep_results)
            if self.with_logprobs:
                postprocessed_logprobs.append(keep_logprobs)

        return postprocessed_preds, postprocessed_logprobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['ft', 'supicl', 'icl'])
    parser.add_argument('--model_version', type=str, default='small', choices=['small', 'base', 'large', 'xl'])
    parser.add_argument('--data', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--train_size', type=int)

    parser.add_argument('--input_format', type=str, default=None)
    parser.add_argument('--model', type=str, default='google/flan-t5')
    parser.add_argument('--train_seeds', type=int, default=42, nargs='+')
    parser.add_argument('--esb_num', type=int)

    parser.add_argument('--use_logprobs', action='store_true')

    # parser.add_argument('--esb_cfg_file', type=str)
    parser.add_argument('--strategy', type=str, default='majority_vote', choices=['majority_vote', 'mean_prob', 'max_prob'])
    parser.add_argument('--keep_distinct', action='store_true')
    parser.add_argument('--min_votes', type=int, default=1)
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--output', type=str, default='outputs')

    parser.add_argument('--ic_num', type=int, default=3)
    parser.add_argument('--train_retrieve', type=str, default='random')
    parser.add_argument('--test_retrieve', type=str, default='random')
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--test_imbalance', action='store_true')

    parser.add_argument('--ablation', action='store_true')

    args = parser.parse_args()

    args.data_cfg = {'task': args.task}
    args.esb_file_dirs = []

    args.esb_file_dirs = dict()

    args.new_run_name = f"zeroshot-{args.task}-{args.ic_num}-{args.train_retrieve}-{args.input_format if args.input_format is not None else 'prompt_cycling'}-{args.strategy}-{args.esb_num}-trainsize_{args.train_size}-{args.model_version}"

    cfg_file = f"cfg/{args.task}/{args.mode}.yaml"
    with open(cfg_file) as f:
        training_cfg = yaml.safe_load(f)

    if 'icl_cfg' in training_cfg.keys():
        args.icl_cfg = training_cfg['icl_cfg']

        if args.train_retrieve is not None:
            args.icl_cfg['retrieve']['train'] = args.train_retrieve
        if args.test_retrieve is not None:
            args.icl_cfg['retrieve']['other'] = args.test_retrieve
        if args.ic_num is not None:
            args.icl_cfg['ic_num'] = args.ic_num
    else:
        args.icl_cfg = None

    args.model_ckpt = None      # note: just to avoid error for run_name
    args.do_train = False        # note: just to obtain the zeroshot run name
    args.with_logprobs = True
    ####################### above is for training, below is for inference ################################

    # this is when you don't have a prompt template and thus using prompt cycling
    train_run_names = [None]

    args.esb_file_dirs = {k: [] for k in train_run_names}

    for train_run_name in train_run_names:
        for prompt_name in os.listdir(f"prompt/{args.task}"):
            prompt_name = prompt_name.split('.json')[0].strip()

            python_command = rf"""python train_ft.py \
                                    --do_predict \
                                    --mode {args.mode} \
                                    --model {args.model} \
                                    --model_version {args.model_version} \
                                    --data {args.data} \
                                    --task {args.task} \
                                    --input_format {prompt_name} \
                                    --train_size {args.train_size} \
                                    --with_logprobs \
                                    --ensemble \
                                    --output {args.output} """
            if args.do_inference:
                os.system(python_command)

            args.input_format = prompt_name
            args.seed = 42
            run_name = create_run_name(args, training_cfg)

            result_dir = f"{args.output}/esb-{run_name}"
            args.esb_file_dirs[train_run_name].append(result_dir)

    ensemble_evaluation_results = dict()

    for i, (train_run_name, esb_file_dirs) in enumerate(args.esb_file_dirs.items()):
        preds = dict()
        for folder in esb_file_dirs:
            preds[folder] = load_result_json(folder)

        golds = preds[esb_file_dirs[0]]

        print('Loaded the prediction files: \n', '\n'.join(esb_file_dirs))
        print('Checking whether the prediction files have the same number of data items...')

        if check_results_num(preds):
            print('Finish checking!')
        else:
            raise AssertionError('Different number of data items identified in the prediction files. ')

        voters = Voters(
            task=args.task,
            pools=preds,
            strategy=args.strategy,
            keep_distinct=args.keep_distinct,
            min_votes=args.min_votes,
            with_logprobs=args.use_logprobs
        )

        print('Voter established! \nStart voting ...')
        postprocessed_results, postprocessed_logprobs = voters.vote()

        assert len(golds) == len(postprocessed_results)

        if args.keep_distinct:
            evaluator = EvaluateTool(args)

            evaluate_results = evaluator.evaluate(
                preds=postprocessed_results,
                golds=golds,
                logprobs=postprocessed_logprobs,
                section=None,
                finish=True,
                ensemble_only=True
            )

            if args.ablation:
                esb_result_dir = os.path.join('ensemble_results', "ablation-"+args.data_cfg['task'])
            else:
                esb_result_dir = os.path.join('ensemble_results', args.data_cfg['task'])
            if not os.path.exists(esb_result_dir):
                os.makedirs(esb_result_dir, exist_ok=True)
            with open(os.path.join(esb_result_dir, f'{args.new_run_name}-seed{args.train_seeds[i]}.json'), 'w') as f:
                json.dump(
                    evaluate_results,
                    f,
                    indent=4
                )
            ensemble_evaluation_results[train_run_name] = evaluate_results
            print(evaluate_results)
        else:
            with open('temp.json', 'w') as f:
                json.dump(
                    [dict(**{"postprocess_prediction": postprocessed_results[idx]}) for idx in range(len(postprocessed_results))],
                    f,
                    indent=4,
                )
            print('Save to temp.json file. ')

    if args.keep_distinct:
        ensemble_results = list(ensemble_evaluation_results.values())
        ks = list(ensemble_results[0].keys())
        final_results = dict()
        for k in ks:
            kv = [i[k] for i in ensemble_results]
            mean_kv = float(np.mean(kv))
            var_kv = float(np.var(kv))
            std_kv = float(np.std(kv))
            final_results[k] = {
                'mean': mean_kv,
                'var': var_kv,
                'std': std_kv
            }

        if args.ablation:
            esb_result_dir = os.path.join('ensemble_inference_results', 'ablation-'+args.data_cfg['task'])
        else:
            esb_result_dir = os.path.join('ensemble_inference_results', args.data_cfg['task'])
        if not os.path.exists(esb_result_dir):
            os.makedirs(esb_result_dir, exist_ok=True)
        with open(os.path.join(esb_result_dir, f'ensemble-{args.new_run_name}.json'), 'w') as f:
            json.dump(
                final_results,
                f,
                indent=4
            )
        print("*"*10, "ensemble over three runs", "*"*10)
        print(final_results)