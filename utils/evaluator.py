# encoding=utf8
import json
import copy
import torch
from typing import DefaultDict
from tkinter import _flatten

import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score, accuracy_score
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError

from utils.verbalizer import VERBALIZER
from utils.batchnorm import BatchNormCalibrate

SST2_LABEL_MAP = {VERBALIZER['sst2'][i]: i for i in range(len(VERBALIZER['sst2']))}

SST5_LABEL_MAP = {VERBALIZER['sst5'][i]: i for i in range(len(VERBALIZER['sst5']))}

MANIFESTOS_LABEL_MAP = {VERBALIZER['manifestos'][i]: i for i in range(len(VERBALIZER['manifestos']))}

INTENTS_LABEL_MAP = {VERBALIZER['intents'][i]: i for i in range(len(VERBALIZER['intents']))}

RTE_LABEL_MAP = {VERBALIZER['rte'][i]: i for i in range(len(VERBALIZER['rte']))}

ANLI_LABEL_MAP = {VERBALIZER['anli'][i]: i for i in range(len(VERBALIZER['anli']))}

HATE_SPEECH_LABEL_MAP = {VERBALIZER['hate_speech'][i]: i for i in range(len(VERBALIZER['hate_speech']))}


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args
        # self.f1 = evaluate.load('f1')
        self.f1 = f1_score
        # self.acc = evaluate.load('accuracy')
        self.acc = accuracy_score
        self.rouge = evaluate.load('rouge')
        self.task = self.args.data_cfg['task']
        # self.domain = self.args.data_cfg['domain']

        with open('utils/ontology.json') as f:
            ontology = json.load(f)

        if self.task == "intents":
            # todo: this has to be the same in the data loading scripts utils/data_wrapper/multi3nlu.py
            # self.intent_desc_dict = {key: ontology["intents"][key]["description"][14:-1] for key in
            #                     ontology["intents"].keys() if
            #                     "general" in ontology["intents"][key]["domain"] or self.domain in
            #                     ontology["intents"][key]["domain"]}
            self.intent_desc_dict = {key: ontology["intents"][key]["description"][14:-1] for key in
                                     ontology["intents"].keys()}
            self.num_intents = len(self.intent_desc_dict)
            self.ece = BinaryCalibrationError(
                n_bins=10, norm='l1'
            )
        elif self.task == "slots":
            # self.intent_desc_dict = {key: ontology["slots"][key]["description"] for key in ontology["slots"].keys() if
            #                     "general" in ontology["slots"][key]["domain"] or self.domain in ontology["slots"][key][
            #                         "domain"]}
            self.intent_desc_dict = {key: ontology["slots"][key]["description"] for key in ontology["slots"].keys()}
            self.num_intents = len(self.intent_desc_dict)
            self.intents_or_slots_list = sorted(list(self.intent_desc_dict.keys()))
        elif self.task in ['sst2', 'rte', 'cause_and_effect', 'causal_judgment']:
            self.ece = BinaryCalibrationError(
                n_bins=10, norm='l1'
            )
        elif self.task in ['sst5']:
            self.ece = MulticlassCalibrationError(
                num_classes=5,
                n_bins=10,
                norm='l1'
            )
        elif self.task in ['anli']:
            self.ece = MulticlassCalibrationError(
                num_classes=3,
                n_bins=10,
                norm='l1'
            )
        elif self.task in ['manifestos']:
            self.ece = MulticlassCalibrationError(
                num_classes=8,
                n_bins=10,
                norm='l1'
            )
        elif self.task in ['hate_speech']:
            self.ece = MulticlassCalibrationError(
                num_classes=3,
                n_bins=10,
                norm='l1'
            )

        if self.args.with_logprobs and self.task in ['sst5', 'sst2', "intents", 'rte', 'anli', 'causal_judgment',
                                                     'cause_and_effect', 'manifestos', 'hate_speech']:
            self.bn = BatchNormCalibrate(
                momentum=0.1
            )

        # self.bleu = evaluate.load('bleu')
        # self.rouge = evaluate.load('rouge')
        # self.bertscore = load_metric('bertscore')
        # self.meteor = load_metric('meteor')

    def evaluate(self, preds, golds, logprobs, section, finish=False, ensemble_only=False):
        if type(logprobs) is not torch.Tensor and logprobs is not None:
            logprobs = torch.tensor(logprobs)

        if self.task in ['slots', 'intents']:
            results = self.evaluate_multi3nlu(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['sst2']:
            results = self.evaluate_sst2(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['sst5']:
            results = self.evaluate_sst5(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['rte', 'cause_and_effect', 'causal_judgment']:
            results = self.evaluate_rte(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['anli']:
            results = self.evaluate_anli(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['manifestos']:
            results = self.evaluate_manifestos(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )
        elif self.task in ['hate_speech']:
            results = self.evaluate_hate_speech(
                preds=preds,
                golds=golds,
                section=section,
                logprobs=logprobs,
                finish=finish,
                ensemble_only=ensemble_only
            )

        return results

    def evaluate_hate_speech(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        gold_labels = list(map(lambda x: HATE_SPEECH_LABEL_MAP[x['labels']], golds))
        pred_labels = list(map(lambda x: HATE_SPEECH_LABEL_MAP[x], preds))
        macro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="macro")
        micro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="micro")
        results['macro_f1'] = float(macro_f1)
        results['micro_f1'] = float(micro_f1)

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
            pred_probs = torch.softmax(logprobs.cpu(), -1)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
            #                                                              target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(
                torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits.cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            bned_macro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="macro")
            results['bn_macro_f1'] = float(bned_macro_f1)
            bned_micro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="micro")
            results['bn_micro_f1'] = float(bned_micro_f1)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results

    def evaluate_sst2(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            gold_labels = list(map(lambda x: SST2_LABEL_MAP[x['labels']], golds))
            ece_results = self.ece(pred_probs[:, SST2_LABEL_MAP['positive']].cpu(), torch.tensor(gold_labels))
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
            pred_probs = torch.softmax(logprobs, -1)
            # pred_probs = pred_probs[:, SST2_LABEL_MAP['positive']]
            gold_labels = list(map(lambda x: SST2_LABEL_MAP[x['labels']], golds))
            ece_results = self.ece(pred_probs[:, SST2_LABEL_MAP['positive']].cpu(), torch.tensor(gold_labels))
            results['ece'] = float(ece_results)

            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs*torch.log(pred_probs), target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits[:, SST2_LABEL_MAP['positive']].cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits*torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results

    def evaluate_rte(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            gold_labels = list(map(lambda x: RTE_LABEL_MAP[x['labels']], golds))
            ece_results = self.ece(pred_probs[:, RTE_LABEL_MAP['yes']].cpu(), torch.tensor(gold_labels))
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
            pred_probs = torch.softmax(logprobs, -1)
            # pred_probs = pred_probs[:, RTE_LABEL_MAP['yes']]
            gold_labels = list(map(lambda x: RTE_LABEL_MAP[x['labels']], golds))
            ece_results = self.ece(pred_probs[:, RTE_LABEL_MAP['yes']].cpu(), torch.tensor(gold_labels))
            results['ece'] = float(ece_results)
            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
            #                                                              target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits[:, RTE_LABEL_MAP['yes']].cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results

    def evaluate_anli(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        gold_labels = list(map(lambda x: ANLI_LABEL_MAP[x['labels']], golds))
        pred_labels = list(map(lambda x: ANLI_LABEL_MAP[x], preds))
        macro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="macro")
        micro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="micro")
        results['macro_f1'] = float(macro_f1)
        results['micro_f1'] = float(micro_f1)

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
            pred_probs = torch.softmax(logprobs.cpu(), -1)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
            #                                                              target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(
                torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits.cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            bned_macro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="macro")
            results['bn_macro_f1'] = float(bned_macro_f1)
            bned_micro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="micro")
            results['bn_micro_f1'] = float(bned_micro_f1)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results

    def evaluate_sst5(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        gold_labels = list(map(lambda x: SST5_LABEL_MAP[x['labels']], golds))
        pred_labels = list(map(lambda x: SST5_LABEL_MAP[x], preds))
        macro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="macro")
        micro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="micro")
        results['macro_f1'] = float(macro_f1)
        results['micro_f1'] = float(micro_f1)

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
            pred_probs = torch.softmax(logprobs.cpu(), -1)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
            #                                                              target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(
                torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits.cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            bned_macro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="macro")
            results['bn_macro_f1'] = float(bned_macro_f1)
            bned_micro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="micro")
            results['bn_micro_f1'] = float(bned_micro_f1)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results

    def evaluate_multi3nlu(self, preds, golds, section, logprobs, finish, ensemble_only):
        gold_labels = [item["labels"] for item in golds]
        pred_tokens = copy.copy(preds)

        # finish=False
        if not finish:
            decoded_preds, decoded_labels = postprocess_text(preds, golds)

            results = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            results = {k: round(v * 100, 4) for k, v in results.items()}
        else:
            gold_texts = [gold_labels[i * self.num_intents: (i + 1) * self.num_intents] for i in
                          range(int(len(gold_labels) / self.num_intents + 1))][:-1]
            preds = [preds[i * self.num_intents:(i * self.num_intents + self.num_intents)] for i in
                     range(int(len(preds) / self.num_intents + 1))][:-1]

            if self.task == 'intents':
                gold_values = [[INTENTS_LABEL_MAP['yes'] if "yes" in int_out else INTENTS_LABEL_MAP['no'] for int_out in output] for output in gold_texts]
                pred_values = [[INTENTS_LABEL_MAP['yes'] if "yes" in int_out else INTENTS_LABEL_MAP['no'] for int_out in output] for output in preds]

            elif self.task == 'slots':
                gold_values = [
                    {slot: label_value for slot, label_value in zip(self.intents_or_slots_list, sent_label) if
                     label_value != "unanswerable"} for sent_label in gold_texts]
                pred_values = [{slot: pred_value for slot, pred_value in zip(self.intents_or_slots_list, sent_output) if
                                pred_value != "unanswerable"} for sent_output in preds]
            else:
                raise ValueError(f'Unsupported task setting: {self.task}. specify this in the data_cfg file. ')

            assert len(pred_values) == len(gold_values)

            if self.task == "intents":
                macro_f1_result = self.f1(
                    y_pred=pred_values,
                    y_true=gold_values,
                    average='macro'
                )
                micro_f1_result = self.f1(
                    y_pred=pred_values,
                    y_true=gold_values,
                    average='micro'
                )
                acc_result = self.acc(
                    y_pred=pred_values,
                    y_true=gold_values
                )
                results = {'overall_macro_f1': float(macro_f1_result),
                           'overall_micro_f1': float(micro_f1_result),
                           'overall_acc': float(acc_result)}
                if logprobs is not None and ensemble_only:
                    golds = list(_flatten(gold_values))
                    pred_probs = torch.tensor(logprobs)
                    results['ece'] = float(self.ece(
                        pred_probs[:len(golds), INTENTS_LABEL_MAP['yes']].cpu(),
                        torch.tensor(golds)
                    ))
                    results['nll'] = float(
                        torch.nn.functional.nll_loss(input=torch.log(pred_probs[:len(golds), :].cpu()),
                                                     target=torch.tensor(golds)))
                    results['mean_p'] = float(
                        torch.mean(torch.tensor([max(x) for x in pred_probs[:len(golds), :].tolist()])))
                    results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

                if logprobs is not None and not ensemble_only:
                    # logprobs = list(map(lambda x: sum(x) / len(x), logprobs))
                    # pred_logprobs = [logprobs[i * self.num_intents:(i * self.num_intents + self.num_intents)] for i in
                    #                  range(int(len(logprobs) / self.num_intents + 1))][:-1]
                    # logprobs = [pred_logprobs[i][j] if pred_values[i][j] == 1 else 1 - pred_logprobs[i][j] for j in
                    #             range(self.num_intents) for i in range(len(pred_logprobs))]
                    golds = list(_flatten(gold_values))
                    pred_probs = torch.softmax(logprobs, -1)
                    results['ece'] = float(self.ece(
                        pred_probs[:len(golds), INTENTS_LABEL_MAP['yes']].cpu(),
                        torch.tensor(golds)
                    ))
                    results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs[:len(golds), :].cpu()), target=torch.tensor(golds)))
                    results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs[:len(golds), :].tolist()])))
                    results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
                    # results['info_entropy'] = float(
                    #     torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
                    #                                  target=torch.tensor(golds)))

                    self.bn.clear()
                    self.bn.train()
                    logits = self.bn(logprobs[:len(golds), :].cpu())
                    self.bn.eval()
                    bned_logits = self.bn(logits.cpu())
                    bned_logits = torch.softmax(bned_logits, -1)
                    bned_preds = torch.argmax(bned_logits, -1)
                    bned_ece_results = self.ece(bned_logits[:, INTENTS_LABEL_MAP['yes']].cpu(),
                                                torch.tensor(golds))
                    results['bn_ece'] = float(bned_ece_results)

                    bned_pred_values = [bned_preds[i * self.num_intents:(i * self.num_intents + self.num_intents)].cpu().tolist()
                                        for i in range(int(len(bned_preds) / self.num_intents))]
                    results['bn_macro_f1'] = self.f1(
                        y_pred=bned_pred_values,
                        y_true=gold_values,
                        average='macro'
                    )
                    results['bn_micro_f1'] = self.f1(
                        y_pred=bned_pred_values,
                        y_true=gold_values,
                        average='micro'
                    )
                    results['bn_acc'] = self.acc(
                        y_pred=bned_pred_values,
                        y_true=gold_values
                    )
                    results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits.cpu()), target=torch.tensor(golds)))
                    results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
                    results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
                    # results['bn_info_entropy'] = float(
                    #     torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
                    #                                  target=torch.tensor(golds)))

                # results = {**f1_result, **acc_result}
            elif self.task == 'slots':
                slot_list = set()
                true_positives = DefaultDict(lambda: 0)
                num_predicted = DefaultDict(lambda: 0)
                num_to_recall = DefaultDict(lambda: 0)
                for output, label in zip(pred_values, gold_values):
                    for slot in output.keys():
                        slot_list.add(slot)
                        num_predicted[slot] += 1

                    for slot in label.keys():
                        slot_list.add(slot)
                        gold_text = label[slot]
                        num_to_recall[slot] += 1
                        if slot in output.keys() and output[slot] == gold_text:
                            true_positives[slot] += 1

                slot_type_f1_scores = DefaultDict()
                slot_type_precision = []
                slot_type_recall = []

                for slot in slot_list:
                    slot_tp, slot_predicted, slot_to_recall = true_positives[slot], num_predicted[slot], num_to_recall[
                        slot]
                    slot_precision = precision(slot_tp, slot_predicted)
                    slot_recall = recall(slot_tp, slot_to_recall)

                    slot_type_precision.append(slot_precision)
                    slot_type_recall.append(slot_recall)
                    slot_type_f1_scores[slot] = calculate_f1(slot_precision, slot_recall)
                    # print(slot, slot_tp, slot_predicted, slot_to_recall, slot_precision, slot_recall)

                averaged_f1 = np.mean(list(slot_type_f1_scores.values()))
                averaged_precision = np.mean(slot_type_precision)
                averaged_recall = np.mean(slot_type_recall)

                overall_true_positives = sum(true_positives.values())
                overall_num_predicted = sum(num_predicted.values())
                overall_num_to_recall = sum(num_to_recall.values())

                overall_precision = precision(overall_true_positives, overall_num_predicted)
                overall_recall = recall(overall_true_positives, overall_num_to_recall)
                overall_f1 = calculate_f1(overall_precision, overall_recall)
                results = {
                    'overall_prec': float(overall_precision),
                    'overall_recall': float(overall_recall),
                    'overall_f1': float(overall_f1),
                    'aver_slottype_prec': float(averaged_precision),
                    'aver_slottype_recall': float(averaged_recall),
                    'aver_slottype_f1': float(averaged_f1)
                }
                if logprobs is not None:
                    probs = list(map(lambda x: x[0], logprobs))
                    # probs = list(map(lambda x: sum(x)/len(x), logprobs))
                    ece = ece_for_qa(preds=pred_tokens, golds=gold_labels, confidences=probs, bin_num=10)
                    results = {**results, **ece}
                    results['mean_p'] = float(torch.mean(torch.tensor(probs)))
            else:
                raise ValueError(f'{self.task} task not supported in evaluator for multi3nlu')

        return results

    def evaluate_manifestos(self, preds, golds, section, logprobs, finish, ensemble_only):
        accuracies = torch.tensor([1 if pred_value.strip() == gold_value['labels'].strip() else 0
                                   for pred_value, gold_value in zip(preds, golds)], dtype=torch.float)
        results = {'acc': float(torch.mean(accuracies))}

        gold_labels = list(map(lambda x: MANIFESTOS_LABEL_MAP[x['labels']], golds))
        pred_labels = [MANIFESTOS_LABEL_MAP[x] if x in MANIFESTOS_LABEL_MAP.keys() else MANIFESTOS_LABEL_MAP['other'] for x in preds]
        # pred_labels = list(map(lambda x: MANIFESTOS_LABEL_MAP[x], preds))
        macro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="macro")
        micro_f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, average="micro")
        results['macro_f1'] = float(macro_f1)
        results['micro_f1'] = float(micro_f1)

        if logprobs is not None and ensemble_only:
            pred_probs = torch.tensor(logprobs)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(
                torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))

        if logprobs is not None and not ensemble_only:
        #     probs = list(map(lambda x: x[0], logprobs))
        #     gold_texts = [x['labels'] for x in golds]
        #     # probs = list(map(lambda x: sum(x)/len(x), logprobs))
        #     ece = ece_for_qa(preds=preds, golds=gold_texts, confidences=probs, bin_num=10)
        #     results = {**results, **ece}
        #     results['mean_p'] = float(torch.mean(torch.tensor(probs)))
        #
        # # if logprobs is not None:
            pred_probs = torch.softmax(logprobs.cpu(), -1)
            ece_results = self.ece(
                pred_probs, torch.tensor(gold_labels)
            )
            results['ece'] = float(ece_results)
            results['nll'] = float(torch.nn.functional.nll_loss(input=torch.log(pred_probs), target=torch.tensor(gold_labels)))
            results['mean_p'] = float(torch.mean(torch.tensor([max(x) for x in pred_probs.tolist()])))
            results['info_entropy'] = -float(torch.mean(pred_probs * torch.log(pred_probs)))
            # results['info_entropy'] = float(torch.nn.functional.nll_loss(input=pred_probs * torch.log(pred_probs),
            #                                                              target=torch.tensor(gold_labels)))

            self.bn.clear()
            self.bn.train()
            logits = self.bn(logprobs.cpu())
            self.bn.eval()
            bned_logits = self.bn(logits.cpu())
            bned_logits = torch.softmax(bned_logits, -1)
            bned_preds = torch.argmax(bned_logits, -1)
            results['bn_acc'] = float(
                torch.mean((bned_preds == torch.tensor(gold_labels).to(bned_preds.device).int()).float()))
            bned_ece_results = self.ece(bned_logits.cpu(), torch.tensor(gold_labels))
            results['bn_ece'] = float(bned_ece_results)
            bned_macro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="macro")
            results['bn_macro_f1'] = float(bned_macro_f1)
            bned_micro_f1 = f1_score(y_true=gold_labels, y_pred=bned_preds, average="micro")
            results['bn_micro_f1'] = float(bned_micro_f1)
            results['bn_nll'] = float(torch.nn.functional.nll_loss(input=torch.log(bned_logits), target=torch.tensor(gold_labels)))
            results['bn_mean_p'] = float(torch.mean(torch.tensor([max(x) for x in bned_logits.tolist()])))
            results['bn_info_entropy'] = -float(torch.mean(bned_logits * torch.log(bned_logits)))
            # results['bn_info_entropy'] = float(torch.nn.functional.nll_loss(input=bned_logits * torch.log(bned_logits),
            #                                                                 target=torch.tensor(gold_labels)))

        return results


def calculate_f1(precision, recall):
    f1_score = 2 * precision * recall / (1e-6 + precision + recall)
    return f1_score


def precision(true_positives, num_predicted):
    return true_positives / (1e-6 + num_predicted)


def recall(true_positives, num_to_recall):
    return true_positives / (1e-6 + num_to_recall)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def ece_for_qa(preds, golds, confidences, bin_num, return_acc=False):
    bin_boundaries = torch.linspace(0, 1, bin_num+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = torch.tensor(confidences)

    accuracies = torch.tensor([1 if pred_value.strip() == gold_value.strip() else 0
                               for pred_value, gold_value in zip(preds, golds)])

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    if return_acc:
        return {'acc': float(torch.mean(accuracies)), 'ece': float(ece)}
    else:
        return {'ece': float(ece)}