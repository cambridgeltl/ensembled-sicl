import argparse
import os
import shutil

import yaml
import json
import numpy as np

from utils.verbalizer import VERBALIZER
from utils.run_config import create_run_name
from utils.evaluator import EvaluateTool
from ensemble_postprocess import load_result_json, check_results_num, Voters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # learning mode arguments
    parser.add_argument('--mode', type=str, default='ft',
                        choices=['ft', 'icl', 'supicl'])
    # model arguments
    parser.add_argument('--model', type=str, default='google/flan-t5',
                        choices=['google/flan-t5', 'meta-llama/Llama-2-7b-hf', 't5'])
    parser.add_argument('--model_version', type=str, default='small', choices=['small', 'base', 'large', 'xl'])
    parser.add_argument('--model_ckpt', type=str, default=None, help='path of the checkpoint')
    parser.add_argument('--load_last_checkpoint', action='store_true')
    # training arguments
    parser.add_argument('--cfg_path', type=str, default='cfg')
    parser.add_argument('--patience', type=int, default=5)
    # data arguments
    parser.add_argument('--data', type=str, default='multi3nlu',
                        choices=['multi3nlu', 'clinc150', 'sst2', 'sst5', 'rte', 'anli', 'causal_judgment',
                                 'cause_and_effect', 'manifestos', 'hate_speech'])
    parser.add_argument('--task', type=str, default='intents',
                        choices=['intents', 'slots', 'sst2', 'sst5', 'rte', 'anli', 'causal_judgment',
                                 'cause_and_effect', 'manifestos', 'hate_speech'])
    parser.add_argument('--train_size', type=str, default=str(50),
                        choices=['16', '50', '100', '200', '300', '500', '800', '1000', '1500', '2000', 'all', '10000',
                                 '30000'])
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--test_imbalance', action='store_true')

    # input format argument
    parser.add_argument('--input_format', type=str, default=None)
    # output configuration
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--report_to', type=str, default="wandb")
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--with_logprobs', action='store_true')

    # train, val, test
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--do_predict', action="store_true")

    # randomness
    parser.add_argument('--train_seeds', type=int, nargs='+', default=[0, 21, 42])
    parser.add_argument("--local-rank", type=int)

    # shortcut customization
    parser.add_argument('--ic_retrieve', type=str, default=None,
                        choices=['random', 'balance', 'sbert', 'bm25', 'label_based'])
    parser.add_argument('--ic_num', type=int, default=None)

    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--keep_distinct', default=True, action='store_true')
    parser.add_argument('--min_votes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='majority_vote',
                        choices=['majority_vote', 'mean_prob', 'max_prob'])
    parser.add_argument('--use_logprobs', action='store_true')

    args = parser.parse_args()

    args.data_cfg = {'task': args.task}

    with open(os.path.join(args.cfg_path, args.task, args.mode + '.yaml')) as f:
        file = f.read()
        training_cfg = yaml.safe_load(file)

    if 'icl_cfg' in training_cfg.keys():
        args.icl_cfg = training_cfg['icl_cfg']

        if args.ic_retrieve is not None:
            args.icl_cfg['retrieve']['train'] = args.ic_retrieve
            args.icl_cfg['retrieve']['other'] = args.ic_retrieve
        if args.ic_num is not None:
            args.icl_cfg['ic_num'] = args.ic_num
    else:
        args.icl_cfg = None

    # Construct the run_name of the task
    args.with_logprobs = True
    train_run_names = []
    for seed in args.train_seeds:
        args.seed = seed
        if args.mode not in ['icl']:
            # this is for supicl, using different input combinations by manipulating different random seeds
            train_run_name = create_run_name(args, training_cfg)
        else:
            train_run_name = None
        train_run_names.append(train_run_name)

    result_files = []
    args.esb_file_dirs = dict()
    if args.mode not in ['icl']:
        args.esb_file_dirs = {k: [] for k in train_run_names}
    else:
        args.esb_file_dirs = {s: [] for s in args.train_seeds}

    args.new_run_name = f"{args.mode}-{args.task}-{args.ic_num}-{args.input_format if args.input_format is not None else 'prompt_cycling'}-{args.strategy}-ALLENSEMBLE"

    for i, train_seed in enumerate(args.train_seeds):
        train_run_name = train_run_names[i]
        for prompt_name in os.listdir(f"prompt/{args.task}"):
            for seed in range(5):
                prompt_name = prompt_name.split('.json')[0].strip()
                args.seed = seed
                args.input_format = prompt_name

                if train_run_name:
                    python_command = rf"""python train_ft.py \
                                --do_predict \
                                --mode {args.mode} \
                                --model {args.model} \
                                --model_version {args.model_version} \
                                --data {args.data} \
                                --task {args.task} \
                                --input_format {prompt_name} \
                                --seed {seed} \
                                --train_size {args.train_size} \
                                --with_logprobs \
                                --ensemble \
                                --output {args.output} \
                                --load_last_checkpoint \
                                --model_ckpt {args.output}/{train_run_name}"""
                else:
                    python_command = rf"""python train_ft.py \
                                            --do_predict \
                                            --mode {args.mode} \
                                            --model {args.model} \
                                            --model_version {args.model_version} \
                                            --data {args.data} \
                                            --task {args.task} \
                                            --input_format {prompt_name} \
                                            --seed {seed} \
                                            --train_size {args.train_size} \
                                            --with_logprobs \
                                            --ensemble \
                                            --output {args.output}"""

                if args.do_inference:
                    os.system(python_command)

                args.with_logprobs = True
                run_name = create_run_name(args, training_cfg)
                result_dir = f"{args.output}/esb-{run_name}"
                result_files.append(os.path.join(result_dir, 'predict_results.json'))

                if args.mode not in ['icl']:
                    if not os.path.exists(result_dir + f"-pc-train_seed{train_seed}"):
                        shutil.copytree(result_dir, result_dir + f"-pc-train_seed{train_seed}")
                    else:
                        if args.do_inference:
                            shutil.rmtree(result_dir + f"-pc-train_seed{train_seed}")
                            os.rename(result_dir,
                                      result_dir + f"-pc-train_seed{train_seed}")

                    args.esb_file_dirs[train_run_name].append(result_dir + f"-pc-train_seed{train_seed}")
                else:
                    args.esb_file_dirs[train_seed].append(result_dir)

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

        evaluator = EvaluateTool(args)

        evaluate_results = evaluator.evaluate(
            preds=postprocessed_results,
            golds=golds,
            logprobs=postprocessed_logprobs,
            section=None,
            finish=True,
            ensemble_only=True
        )

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
    esb_result_dir = os.path.join('ensemble_inference_results', args.data_cfg['task'])
    if not os.path.exists(esb_result_dir):
        os.makedirs(esb_result_dir, exist_ok=True)
    with open(os.path.join(esb_result_dir, f'ensemble-{args.new_run_name}.json'), 'w') as f:
        json.dump(
            final_results,
            f,
            indent=4
        )
    print("*" * 10, "ensemble over three runs", "*" * 10)
    print(final_results)