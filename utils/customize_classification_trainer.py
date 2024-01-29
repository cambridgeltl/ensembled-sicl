import collections
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import NamedTuple
from torch.nn import CrossEntropyLoss

import datasets
import numpy as np
import torch
from torch import nn
import transformers.trainer_seq2seq
from torch.utils.data import Dataset
from packaging import version
from transformers.trainer_utils import PredictionOutput, speed_metrics
from torch.utils.data import DataLoader, Dataset
from transformers.deepspeed import deepspeed_init

from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_torch_tpu_available,
    logging,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled

from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from utils.verbalizer import VERBALIZER
from utils.batchnorm import BatchNormCalibrate

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]


class EvaluateFriendlyClassificationTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            weighted_logprobs: bool = False,
            task: str = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir
        self.weighted_logprobs = weighted_logprobs
        self.task = task
        self.verbalizers = VERBALIZER[self.task]
        self.verbalizer_ids = [self.tokenizer(v, add_special_tokens=False).input_ids for v in self.verbalizers]

        self.bn = BatchNormCalibrate()

    '''def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            raise ValueError("Incompatible with curriculum learning")
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                return SequentialSampler(self.train_dataset)  # Sequential
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                raise ValueError("Incompatible with curriculum learning")
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=False,  # Sequential
                    seed=self.args.seed,
                )'''

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            eval_examples: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # print([eval_examples[idx]['arg_path'] for idx in range(len(eval_examples))])

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output, logprobs, class_logits = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_examples = eval_dataset.dataset.add_column('input_str', eval_dataset.input_record)

            eval_preds = self._post_process_function(
                eval_examples,
                output.predictions,
                "eval_{}".format(self.state.epoch),
                logprob=logprobs if self.weighted_logprobs else None,
                class_logits=class_logits if self.weighted_logprobs else None
            )
            # todo: here we feed the class logits to compute the BN
            summary = self.compute_metrics(eval_preds, logprobs=class_logits, section="dev", finish=True)
            output.metrics.update(summary)

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output, logprobs, class_logits = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            test_examples = test_dataset.dataset.add_column('input_str', test_dataset.input_record)

            eval_preds = self._post_process_function(
                test_examples,
                output.predictions,
                metric_key_prefix,
                logprob=logprobs if self.weighted_logprobs else None,
                class_logits=class_logits if self.weighted_logprobs else None
            )
            # todo: here we feed the class_logits to compute the bn and ece
            output.metrics.update(self.compute_metrics(eval_preds, logprobs=class_logits, section="test", finish=True))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def _post_process_function(
            self, examples: Dataset, predictions: np.ndarray, stage: str, logprob: torch.Tensor = None, class_logits: torch.Tensor = None
    ) -> EvalPrediction:
        # assert isinstance(examples, Dataset)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Save locally.
        if self.args.local_rank <= 0:
            if self.weighted_logprobs:
                with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                    json.dump(
                        [dict(**{"prediction": predictions[idx],
                                 "logprob": str(logprob[idx][logprob[idx].nonzero(as_tuple=True)].tolist()),
                                 'logits': str(class_logits[idx][class_logits[idx].nonzero(as_tuple=True)].tolist())
                                 },
                              **examples[idx]) for idx in range(len(predictions))],
                        f,
                        indent=4,
                    )
            else:
                with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                    json.dump(
                        [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
                        f,
                        indent=4,
                    )

        # Save to wandb.
        if self.wandb_run_dir and self.args.local_rank <= 0:
            if self.weighted_logprobs:
                with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                    json.dump(
                        [dict(**{"prediction": predictions[idx],
                                 "logprob": str(logprob[idx][logprob[idx].nonzero(as_tuple=True)].tolist()),
                                 'logits': str(class_logits[idx][class_logits[idx].nonzero(as_tuple=True)].tolist())
                                 },
                              **examples[idx]) for idx in range(len(predictions))],
                        f,
                        indent=4,
                    )
            else:
                with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                    json.dump(
                        [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
                        f,
                        indent=4,
                    )
        return EvalPrediction(predictions=predictions, items=[examples[idx] for idx in range(len(predictions))])

    def _compute_metrics(self, eval_prediction: EvalPrediction, section, logprobs, finish=False) -> dict:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, logprobs, section, finish=finish)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        # todo:
        logprobs_host = None
        class_logits_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.
        # todo:
        all_logprobs = None
        all_class_logits = None

        observed_num_examples = 0
        logprob_list = []
        logits_list = []
        # verbalizer_max_len = max([len(i) for i in verbalizer_ids])
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # todo: adds for llama
            if self.model.config.is_encoder_decoder:
                inputs['decoder_input_ids'] = (
                        torch.ones((len(inputs['input_ids']), 1)) * torch.tensor(self.tokenizer.pad_token_id)
                ).int()

            # Prediction step
            loss, logits, labels, logprobs, class_logits = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                # verbalizer_ids=self.verbalizer_ids
            )
            logprob_list.append(logprobs)
            logits_list.append(class_logits)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            # todo:
            if logprobs is not None:
                logprobs = self.accelerator.gather_for_metrics((logprobs))
                logprobs_host = logprobs if logprobs_host is None else nested_concat(logprobs_host, logprobs, padding_index=-100)
            if class_logits is not None:
                class_logits = self.accelerator.gather_for_metrics((class_logits))
                class_logits_host = class_logits if class_logits_host is None else nested_concat(class_logits_host, class_logits, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # todo:
                if logprobs_host is not None:
                    logprobs = nested_numpify(logprobs_host)
                    all_logprobs = (
                        logprobs if all_logprobs is None else nested_concat(all_logprobs, logprobs, padding_index=-100)
                    )
                if class_logits_host is not None:
                    class_logits = nested_numpify(class_logits_host)
                    all_class_logits = (
                        class_logits if all_class_logits is None else nested_concat(all_class_logits, class_logits, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                # todo:
                logprobs_host = None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # todo:
        if logprobs_host is not None:
            logprobs = nested_numpify(logprobs_host)
            all_logprobs = logprobs if all_logprobs is None else nested_concat(all_logprobs, logprobs, padding_index=-100)
        if class_logits_host is not None:
            class_logits = nested_numpify(class_logits_host)
            all_class_logits = class_logits if all_class_logits is None else nested_concat(all_class_logits, class_logits,
                                                                               padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples), torch.tensor(all_logprobs), torch.tensor(all_class_logits)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        # verbalizer_ids=None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        verbalizer_max_len = max([len(i) for i in self.verbalizer_ids])
        # if not self.args.predict_with_generate or prediction_loss_only:
        #     return super().prediction_step(
        #         model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        #     )

        # todo:
        if 'labels' in inputs.keys():
            labels = inputs.pop("labels")
        else:
            labels = None

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        # if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
        #     gen_kwargs = self._gen_kwargs.copy()
        #
        # if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
        #     gen_kwargs["max_length"] = self.model.config.max_length
        # gen_kwargs["num_beams"] = (
        #     gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        # )
        # default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        # gen_kwargs["synced_gpus"] = (
        #     gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        # )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        # generated_tokens = self.model.generate(**inputs, **gen_kwargs)

        # todo: adds for llama
        if not self.model.config.is_encoder_decoder and 'total_seq' in inputs.keys():
            total_seq = inputs.pop('total_seq')
            total_seq_attention_mask = inputs.pop('total_seq_attention_mask')

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            class_logits = []
            for verbalizer in self.verbalizer_ids:       # verbalizer_ids: [[182, 1465], [1465], [7163], [2841], [182, 2841]]
                verbalizer_prob = []
                for seq_idx, idx in enumerate(verbalizer):
                    if self.model.config.is_encoder_decoder:
                        token_logits = outputs.logits[:, seq_idx, idx]
                    else:
                        # fixme: only allows verbalizer length to 1
                        token_logits = outputs.logits[:, -1, idx]
                    verbalizer_prob.append(token_logits)
                class_logits.append(torch.mean(torch.stack(verbalizer_prob, dim=0), dim=0))

            class_logits = torch.stack(class_logits, dim=0).T       # shape: [batch_size, num_of_classes]
            class_probs = torch.softmax(class_logits, -1)
            predicted_label = torch.argmax(class_probs, -1)         # shape: [batch_size], eg: [3, 4, 1, 3]
            generated_tokens = torch.tensor(                        # shape: [batch_size, output_max_length]
                [self.verbalizer_ids[i] + [self.tokenizer.eos_token_id] + (verbalizer_max_len - len(self.verbalizer_ids[i])) * [self.tokenizer.pad_token_id]
                 if len(self.verbalizer_ids[i]) < verbalizer_max_len else
                 self.verbalizer_ids[i] + [self.tokenizer.eos_token_id] for i in predicted_label]).to(self.args.device)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        # todo:
        if labels is not None:
            has_labels = True
            inputs['labels'] = labels

        with torch.no_grad():
            criterion = CrossEntropyLoss()
            inputs['labels'][inputs['labels'] == -100] = self.tokenizer.pad_token_id
            targets = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)
            inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
            target_ids = torch.tensor([self.verbalizers.index(t) for t in targets]).int()
            loss = criterion(class_probs, target_ids.long().to(class_probs.device))

            # if has_labels:
            #     if self.label_smoother is not None:
            #         loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
            #     else:
            #         loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            # else:
            #     loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels, class_probs, class_logits

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #
    #     Subclass and override for custom behavior.
    #     """
    #     labels = inputs.pop('labels')
    #     outputs = model(**inputs)
    #
    #     class_logits = []
    #     for verbalizer in self.verbalizer_ids:  # verbalizer_ids: [[182, 1465], [1465], [7163], [2841], [182, 2841]]
    #         verbalizer_prob = []
    #         for seq_idx, idx in enumerate(verbalizer):
    #             token_logits = outputs.logits[:, seq_idx, idx]
    #             verbalizer_prob.append(token_logits)
    #         class_logits.append(torch.mean(torch.stack(verbalizer_prob, dim=0), dim=0))
    #
    #     class_logits = torch.stack(class_logits, dim=0).T  # shape: [batch_size, num_of_classes]
    #     class_probs = torch.softmax(class_logits, -1)
    #
    #     criterion = CrossEntropyLoss()
    #     labels[labels == -100] = self.tokenizer.pad_token_id
    #     targets = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     target_ids = torch.tensor([self.verbalizers.index(t) for t in targets]).int()
    #     loss = criterion(class_probs, target_ids.long().to(class_probs.device))
    #
    #     return (loss, outputs) if return_outputs else loss
