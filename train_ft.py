import os
import wandb
import torch
import logging
import argparse
import yaml
import configparser
import transformers

from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, set_seed, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from utils.run_config import create_run_name
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from utils.load_data import load_data, tokenize_dataset
from utils.load_model import load_model
from utils.evaluator import EvaluateTool
from utils.customize_seq2seq_trainer import EvaluateFriendlySeq2SeqTrainer
from utils.customize_collator import CustomizedCollatorforClassification

logger = logging.getLogger(__name__)

TEMPLATE = {
    'none': 'none_none_none',
    'sqatin': 'usersaid_QUESTION_none'
}

# TODO: setup your WANDB_API_KEY and WANDB_ENTITY here
WANDB_API_KEY = ""
WANDB_ENTITY = ""

def init(args):
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    torch.use_deterministic_algorithms(True)
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    # Read in the training arguments
    # the training argument should be identical for each different learning methods (ft, supicl, ...)
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

    if args.train_bz:
        training_cfg['hyper']['train_batch_size'] = args.train_bz
    if args.val_bz:
        training_cfg['hyper']['val_batch_size'] = args.val_bz
    if args.grad_acc:
        training_cfg['hyper']['grad_accumulation'] = args.grad_acc

    # Construct the run_name of the task
    args.run_name = create_run_name(args, training_cfg)
    if args.ensemble:
        args.run_name = 'esb-' + args.run_name

    if args.input_format is not None:
        args.input_format = [args.input_format]

    # Initialize the training arguments
    # todo: newly define training arguments
    if args.mode in ['icl']:
        sup_hyper = None
    else:
        sup_hyper = training_cfg['hyper']

    if args.model.lower().startswith('meta-llama'):
        from utils.peft_scripts import create_peft_config
        args.peft_config = create_peft_config(training_cfg['peft_config'])
    elif args.model.lower().startswith('google') and args.model_version == 'xl':
        from utils.peft_scripts import create_peft_config
        training_cfg['peft_config']['target_modules'] = ["q", "v"]
        from peft import (
            LoraConfig,
            TaskType
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=training_cfg['peft_config']['inference_mode'],
            r=int(training_cfg['peft_config']['r']),
            lora_alpha=int(training_cfg['peft_config']['lora_alpha']),
            lora_dropout=float(training_cfg['peft_config']['lora_dropout']),
            target_modules=training_cfg['peft_config']['target_modules']
        )
        args.peft_config = peft_config
    else:
        args.peft_config = None

    training_args = WrappedSeq2SeqTrainingArguments(
        output_dir=os.path.join(args.output, args.run_name),
        remove_unused_columns=False,
        evaluation_strategy=training_cfg['eval']['eval_strategy'],
        eval_steps=training_cfg['eval']['eval_steps'] if training_cfg['eval']['eval_strategy'] == "steps" else None,
        save_strategy=training_cfg['save']['save_strategy'],
        save_steps=training_cfg['save']['save_steps'] if training_cfg['save']['save_strategy'] == "steps" else None,
        save_total_limit=1,
        seed=args.seed,
        # note: only for ft and supicl and supicl_esb
        #############################
        learning_rate=sup_hyper['lr'] if sup_hyper else 0,
        per_device_train_batch_size=sup_hyper['train_batch_size'] if sup_hyper else 0,
        gradient_accumulation_steps=sup_hyper['grad_accumulation'] if sup_hyper else 0,
        per_device_eval_batch_size=sup_hyper['val_batch_size'] if sup_hyper else training_cfg['icl_hyper']['val_batch_size'],
        num_train_epochs=sup_hyper['epochs'] if sup_hyper else 0,
        #############################
        # warmup_ratio=0.1,
        logging_steps=training_cfg['logging']['logging_step'],
        load_best_model_at_end=True,
        metric_for_best_model=training_cfg['eval']['metric'],
        push_to_hub=False,
        # customize
        predict_with_generate=training_cfg['model']['predict_with_generate'],
        generation_max_length=training_cfg['model']['generation_max_length'],
        generation_num_beams=training_cfg['model']['generation_num_beams']
    )

    # Initialize the wandb logger if specified
    if args.report_to == "wandb":
        import wandb
        init_args = {}

        # note: my new wandb api key
        wandb.login(key=WANDB_API_KEY)

        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        run_name = f'supicl_{args.task}'
        if args.ensemble:
            run_name = 'esb_' + run_name
        if args.model_version == 'large':
            run_name = f'large_{run_name}'
        if args.imbalance or args.test_imbalance:
            run_name = f"imb_{run_name}"
        if args.local_rank == 0 or args.local_rank is None:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", run_name),
                name=args.run_name,
                entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
                **init_args,
            )
            wandb.config.update(training_args, allow_val_change=True)
    else:
        training_args.report_to = [None]

    args.run_name = run_name

    # Detect the checkpoint
    # todo: detect the checkpoint
    if args.load_last_checkpoint and args.model_ckpt is not None:
        training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    elif args.load_last_checkpoint and args.model_ckpt is None:
        raise ValueError("Should specify the folder of the load checkpoint.")
    else:
        training_args.load_weights_from = None

    return training_args


if __name__ == '__main__':
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
                        choices=['multi3nlu', 'clinc150', 'sst2', 'sst5', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos', 'hate_speech'])
    parser.add_argument('--task', type=str, default='intents',
                        choices=['intents', 'slots', 'sst2', 'sst5', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos', 'hate_speech'])
    parser.add_argument('--train_size', type=str, default=str(50),
                        choices=['16', '50', '100', '200', '300', '500', '800', '1000', '1500', '2000', 'all', '10000', '30000'])
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--test_imbalance', action='store_true')

    ###########################################
    # parser.add_argument('--domain', type=str, default='banking',
    #                     choices=['banking', 'hotel'])
    # parser.add_argument('--template', type=str, default='none',
    #                     choices=['none', 'sqatin'])
    # parser.add_argument('--file_id', type=int, default=0)
    # parser.add_argument('--train_test_split', type=int, default=10)
    ##########################################


    # input format argument
    parser.add_argument('--input_format', type=str, default=None)
                        # choices=['none', 'sqatin', 'sst2', 'sst5', 'new_1', 'new_2', 'llama_sst2', 'llama_sst5', 'rte',
                        #          'anli', 'causal_judgment', 'cause_and_effect', 'manifestos', 'cycle_1'])
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local-rank", type=int)

    # debug
    parser.add_argument('--toy', action='store_true')

    # shortcut customization
    parser.add_argument('--ic_retrieve', type=str, default=None,
                        choices=['random', 'balance', 'sbert', 'bm25', 'label_based'])
    parser.add_argument('--ic_num', type=int, default=None)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--train_bz', type=int, default=None)
    parser.add_argument('--val_bz', type=int, default=None)
    parser.add_argument('--grad_acc', type=int, default=None)

    args = parser.parse_args()

    training_args = init(args)

    args.data_cfg = dict()
    args.data_cfg['task'] = args.task
    args.data_cfg['train_size'] = args.train_size

    # args.data_cfg['domain'] = args.domain
    # args.data_cfg['train_file'] = f"train_{args.file_id}_{TEMPLATE[args.template]}_{args.train_test_split}_{args.domain}_{args.task}.json"
    # args.data_cfg['test_file'] = args.data_cfg['train_file'].replace('train', 'test')

    print(f'Preparing the {args.data} dataset... ')
    data = load_data(dataset_name=args.data, data_args=args.data_cfg, imbalance=args.imbalance, test_imbalance=args.test_imbalance)

    if len(data) == 2:
        train_split, eval_split, test_split = data['train'], None, data['test']
    else:
        try:
            train_split, eval_split, test_split = data['train'], data['dev'], data['test']
        except:
            train_split, eval_split, test_split = data['train'], data['validation'], data['test']

    if args.toy:
        print('Only using toy examples for debugging...')
        train_split = train_split.select(list(range(10)))
        if eval_split:
            eval_split = eval_split.select(list(range(4)))
        test_split = test_split.select(list(range(4)))

    model, tokenizer = load_model(args)

    tokenized_data, max_source_length, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        processor=tokenizer,
        mode=args.mode,
        input_format=args.input_format,
        # kwargs
        icl_cfg=args.icl_cfg,
        ic_pools=data,
        task=args.task,
        model_type=args.model,
        # randomness in kwargs
        seed=args.seed
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience)
    label_pad_token_id = -100
    # Data collator
    if args.model.startswith('google/flan-t5') or args.model.startswith('t5'):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )
    else:
        # todo: adds for llama
        data_collator = CustomizedCollatorforClassification(
            tokenizer=tokenizer,
            mlm=False
        )
        model.config.max_length = max_source_length + max_target_length

    if args.task in ['sst2', 'sst5', 'intents', 'rte', 'anli', 'causal_judgment', 'cause_and_effect', 'manifestos', 'hate_speech'] and args.with_logprobs:
        from utils.customize_classification_trainer import EvaluateFriendlyClassificationTrainer
        print('Using customized classification trainer', '*' * 20)
        trainer_type = EvaluateFriendlyClassificationTrainer
    else:
        trainer_type = EvaluateFriendlySeq2SeqTrainer

    trainer = trainer_type(
        args=training_args,
        model=model,
        evaluator=EvaluateTool(args=args),
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['eval'] if 'eval' in tokenized_data.keys() else tokenized_data['test'],
        eval_examples=eval_split if 'eval' in tokenized_data.keys() else test_split,
        wandb_run_dir=wandb.run.dir if "wandb" in training_args.report_to and training_args.local_rank <= 0 else None,
        callbacks=[early_stopping_callback],
        weighted_logprobs=args.with_logprobs,
        task=args.task
    )
    print('Trainer build successfully.')

    checkpoint = None
    if training_args.load_weights_from is not None:
        checkpoint = training_args.load_weights_from

    if args.load_last_checkpoint:
        assert args.model_ckpt is not None, ValueError("Should specify the folder of the load checkpoint.")
        assert os.path.exists(args.model_ckpt)
        training_args.load_weights_from = args.model_ckpt
        if training_args.load_weights_from is not None:
            if args.model_version != 'xl':
                state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME),
                                        map_location="cpu")
                trainer.model.load_state_dict(state_dict, strict=True)
                del state_dict
            else:
                from peft import PeftModel, PeftConfig
                from transformers import T5ForConditionalGeneration
                config = PeftConfig.from_pretrained(args.model_ckpt)
                model = T5ForConditionalGeneration.from_pretrained(
                    config.base_model_name_or_path,
                    cache_dir=args.cache_dir
                )
                model.config.use_cache = False
                trainer.model = PeftModel.from_pretrained(model, args.model_ckpt)
                print(f'Loaded PEFT tuned model from {args.model_ckpt}. ')


    # NOTE: train the model with supervision
    if args.mode in ['ft', 'supicl'] and args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(tokenized_data['train'])
        metrics["train_samples"] = min(max_train_samples, len(tokenized_data['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if args.model_version == 'xl':
            trainer.model.save_pretrained(args.run_name)

    if args.do_eval and eval_split is not None:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval"
        )
        max_eval_samples = len(tokenized_data['eval'])
        metrics["eval_samples"] = min(max_eval_samples, len(tokenized_data['eval']))

        # _ = metrics.pop("eval_precisions")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=tokenized_data['test'],
            test_examples=tokenized_data['test'].dataset,
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(tokenized_data['test'])
        metrics["predict_samples"] = min(max_predict_samples, len(tokenized_data['test']))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)