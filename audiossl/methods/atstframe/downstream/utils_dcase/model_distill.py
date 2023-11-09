import torch
import yaml
import os

import pandas as pd
import numpy as np

from pathlib import Path
from torch import nn
from pytorch_lightning import LightningModule

from torch.nn import functional as F
import torchmetrics as tm
from audiossl.datasets.dcase_utils import ManyHotEncoder
from audiossl.utils.common import cosine_scheduler_epoch
from audiossl.methods.atstframe.downstream.utils_dcase.class_dict import classes_labels
from audiossl.methods.atstframe.downstream.utils_psds_eval.gpu_decode import (
    batched_decode_preds,
    log_sedeval_metrics,
    decode_preds,
    MedianPool2d,
    SEDMetrics,

)
from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points
)

'''
This file is modified from model.py
'''


def binary_cross_entropy_with_logits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
    return F.binary_cross_entropy_with_logits(x, y.float())

class LinearHead(nn.Module):

    """Linear layer with attention module for DCASE task"""
    def __init__(self, dim, num_labels=1000,use_norm=True,affine=False):
        super().__init__()
        self.num_labels = num_labels
        self.use_norm=use_norm
        if use_norm:
            self.norm = nn.BatchNorm2d(dim,affine=affine)
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.linear_softmax = nn.Linear(dim, num_labels)
        self.linear_softmax.weight.data.normal_(mean=0.0, std=0.01)
        self.linear_softmax.bias.data.zero_()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, temp=1):
        # flatten
        x = x.transpose(1, 2)
        if self.use_norm:
            x = x.unsqueeze(-1)
            x = self.norm(x)
        x = x.squeeze(-1).transpose(1, 2)
        # linear layer + get strong predictions
        strong_logits = self.linear(x)
        strong = self.sigmoid(strong_logits / temp)

        ### weak logits generated after linear softmax!!!
        weak_logits = self.linear_softmax(x)

        # linear layer + get weak predictions
        soft = self.softmax(weak_logits).clamp(min=1e-7, max=1)
        weak = (strong * soft).sum(1) / soft.sum(1)

        return strong.transpose(1, 2), weak


class DistillPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 dcase_conf,
                 max_epochs,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 n_last_blocks=1,
                 multi_label=False,
                 mixup_training=False,
                 metric_save_dir=None,
                 freeze_mode=False,
                 distill_mode="clip->frame"):
        super().__init__()
        self.freeze_mode = freeze_mode
        self.learning_rate = learning_rate
        print("Using learning rate:", learning_rate)
        self.max_epochs = max_epochs
        self.warumup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch
        self.metric_save_dir = metric_save_dir
        self.encoder = encoder
        self.head = LinearHead(encoder.embed_dim, num_labels, use_norm=False, affine=False)
        self.multi_label = multi_label
        self.mixup_training = mixup_training
        self.num_labels = num_labels
        self.loss_fn = torch.nn.BCELoss()
        self.monitor = 0
        self.mylr_scheduler = cosine_scheduler_epoch(learning_rate,
                                                     1e-6,
                                                     max_epochs,
                                                     niter_per_epoch,
                                                     warmup_epochs)

        self.save_hyperparameters(ignore=["encoder", ])
        self.distill_mode = distill_mode
        with open(dcase_conf, "r") as f:
            self.config = yaml.safe_load(f)
        self.pred_decoder = ManyHotEncoder(
            list(classes_labels.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )
        # for weak labels we simply compute f1 score
        self.get_weak_student_f1_seg_macro = tm.F1Score(
            task="multilabel",
            num_labels=len(self.pred_decoder.labels),
            average="macro",
            compute_on_step=False,
        )
        self.get_weak_teacher_f1_seg_macro = tm.F1Score(
            task="multilabel",
            num_labels=len(self.pred_decoder.labels),
            average="macro",
            compute_on_step=False,
        )

        # buffer for event based scores which we compute using sed-eval
        self.median_filter = MedianPool2d(self.config["training"]["median_window"], same=True)
        self.sed_metrics_student = SEDMetrics(intersection_thd=0.5)
        self.sed_metrics_teacher = SEDMetrics(intersection_thd=0.5)
        
        test_n_thresholds = self.config["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()
        self.ce_loss = nn.CrossEntropyLoss()


    def training_step(self, batch, batch_idx):
        bsz = self.config["training"]["batch_size"]
        indx_synth, indx_weak = bsz
        if not self.freeze_mode:
            self.encoder.finetune_mannual_train()
        self.schedule()
        data, labels, _ = batch

        x, labels = self.encoder((data, labels))
        if self.distill_mode == "clip->frame":
            with torch.no_grad():
                strong_pred_tea, weak_pred_tea = self.encoder.encoder.teacher_module((data, labels))
            strong_pred_std, weak_pred_std = self.head(x)
            
        elif self.distill_mode == "frame->clip":
            with torch.no_grad():
                strong_pred_tea, weak_pred_tea = self.encoder.encoder.teacher_module((data, labels))
            strong_pred_std, weak_pred_std = self.head(x)
        
        loss_d_strong = self.loss_fn(strong_pred_std, strong_pred_tea.detach())
        loss_d_weak = self.loss_fn(weak_pred_std, weak_pred_tea.detach())
        loss_d = (loss_d_strong + loss_d_weak) / 2
        # Distillation loss

        # Get weak and strong mask for two types of data
        strong_mask = torch.zeros(indx_synth + indx_weak).to(x).bool()
        weak_mask = torch.zeros(indx_synth + indx_weak).to(x).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1

        # Get weak label for real data
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        weak_loss = self.loss_fn(weak_pred_std[weak_mask], labels_weak)
        strong_loss = self.loss_fn(strong_pred_std[strong_mask], labels[strong_mask])

        tot_loss = weak_loss + strong_loss

        tot_loss = tot_loss / 2 + loss_d / 2
        
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        self.log("train/real/weak_loss", weak_loss)
        self.log("train/synth/strong_loss", strong_loss)
        self.log("train/total_loss", tot_loss)
        self.log("train/distill_loss", loss_d)

        return tot_loss

    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            param_group["lr"] = self.mylr_scheduler[self.global_step]
            # param_group["lr"] = self.learning_rate
        self.log("lr", param_group["lr"], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        data, labels, filenames = batch

        x, labels = self.encoder((data, labels))
        
        teacher_strong, teacher_weak = self.encoder.encoder.teacher_module((data, labels))
        strong_pred, weak_pred = self.head(x)
        # Get weak and strong mask for two types of data
        mask_weak = (
            torch.tensor([str(Path(x).parent) == str(Path(self.config["data"]["weak_folder"])) for x in filenames])
            .to(x)
            .bool()
        )
        mask_synth = (
            torch.tensor([str(Path(x).parent) == str(Path(self.config["data"]["synth_val_folder"])) for x in filenames])
            .to(x)
            .bool()
        )

        if torch.any(mask_weak):
            labels_weak = (torch.sum(labels[mask_weak], -1) >= 1).float()
            loss_weak_student = self.loss_fn(
                weak_pred[mask_weak], labels_weak
            )
            loss_weak_teacher = self.loss_fn(
                teacher_weak[mask_weak], labels_weak
            )
            self.log("val/weak/teacher/loss_weak", loss_weak_teacher)
            self.log("val/weak/student/loss_weak", loss_weak_student)
            # accumulate f1 score for weak labels
            self.get_weak_student_f1_seg_macro(
                weak_pred[mask_weak], labels_weak.int()
            )
            self.get_weak_teacher_f1_seg_macro(
                teacher_weak[mask_weak], labels_weak.int()
            )
            
        if torch.any(mask_synth):
            loss_strong_student = self.loss_fn(
                strong_pred[mask_synth], labels[mask_synth]
            )
            loss_strong_teacher = self.loss_fn(
                teacher_strong[mask_synth], labels[mask_synth]
            )
            self.log("val/synth/student/loss_strong", loss_strong_student)
            self.log("val/synth/teacher/loss_strong", loss_strong_teacher)
            decoded_student_strong = decode_preds(strong_pred[mask_synth], self.config["training"]["val_thresholds"], self.median_filter)
            decoded_teacher_strong = decode_preds(teacher_strong[mask_synth], self.config["training"]["val_thresholds"], self.median_filter)
                        
            self.sed_metrics_student.accm_macro_f1(decoded_student_strong, labels[mask_synth])
            self.sed_metrics_teacher.accm_macro_f1(decoded_teacher_strong, labels[mask_synth])


    def on_validation_epoch_end(self):
        weak_student_f1_macro = self.get_weak_student_f1_seg_macro.compute()
        weak_teacher_f1_macro = self.get_weak_teacher_f1_seg_macro.compute()
        # synth dataset
        intersection_f1_macro_student = self.sed_metrics_student.compute_macro_f1()
        intersection_f1_macro_teacher = self.sed_metrics_teacher.compute_macro_f1()
        obj_metric = torch.tensor(weak_student_f1_macro.item() + intersection_f1_macro_student.item())

        self.log("val/object_metric", obj_metric, prog_bar=True)
        self.log("val/weak/student/macro_F1", weak_student_f1_macro, on_epoch=True)
        self.log("val/weak/teacher/macro_F1", weak_teacher_f1_macro, on_epoch=True)
        self.log("val/synth/teacher/intersection_f1_macro", intersection_f1_macro_teacher, on_epoch=True)
        self.log("val/synth/student/intersection_f1_macro", intersection_f1_macro_student, on_epoch=True)

        self.get_weak_student_f1_seg_macro.reset()
        self.get_weak_teacher_f1_seg_macro.reset()
        return obj_metric

    def test_step(self, batch, batch_idx):
        self.encoder.eval()

        data, labels, filenames = batch
        x, labels = self.encoder((data, labels))
        
        if self.distill_mode == "clip-->frame":
            teacher_strong, teacher_weak = self.encoder.encoder.teacher_module((data, labels))

        strong_pred, weak_pred = self.head(x)

        # Get weak label for real data
        test_loss = self.loss_fn(strong_pred, labels)
        
        self.log("test/real/strong_loss", test_loss, prog_bar=True, logger=True)
        # Compute PSDS (Different from F1 metric, PSDS computes the ROC, which requires various thresholds from 0 to 1.)
        decoded_strong = batched_decode_preds(
            strong_pred,
            filenames,
            self.pred_decoder,
            median_filter=self.config["training"]["median_window"],
            thresholds=list(self.test_psds_buffer.keys()),
        )
        for th in self.test_psds_buffer.keys():
            self.test_psds_buffer[th] = pd.concat([self.test_psds_buffer[th], decoded_strong[th]], ignore_index=True)

        # Compute F1 metric
        decoded_strong = batched_decode_preds(
            strong_pred,
            filenames,
            self.pred_decoder,
            median_filter=self.config["training"]["median_window"],
            thresholds=[0.5],
        )
        self.decoded_05_buffer = self.decoded_05_buffer.append(decoded_strong[0.5])

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        
        return

    def on_test_epoch_end(self) -> None:
        save_dir = os.path.join(self.metric_save_dir, "metrics_test")

        # calculate the metrics
        psds_score_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer,
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "scenario1"),
        )

        psds_score_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer,
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=os.path.join(save_dir, "scenario2"),
        )

        event_F1_macro = log_sedeval_metrics(
            self.decoded_05_buffer,
            self.config["data"]["test_tsv"],
            os.path.join(save_dir, "student"),
        )[0]

        # synth dataset
        intersection_f1_macro = compute_per_intersection_macro_f1(
            {"0.5": self.decoded_05_buffer},
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
        )

        best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2))

        results = {
            "hp_metric": best_test_result,
            "test/real/psds_score_scenario1": psds_score_scenario1,
            "test/real/psds_score_scenario2": psds_score_scenario2,
            "test/real/event_f1_macro": event_F1_macro,
            "test/real/intersection_f1_macro": intersection_f1_macro,
        }

        if self.logger is not None:
                self.logger.log_metrics(results)
                self.logger.log_hyperparams(self.config, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        # Freezing opt
        if self.freeze_mode:
            optimizer = torch.optim.SGD(
                list(self.head.parameters()),
                self.learning_rate,
                momentum=0.9,
                weight_decay=0,
                )
        # Finetune opt
        else:
            if self.distill_mode == "frame->clip":
                optimizer = torch.optim.SGD(
                    list(self.encoder.encoder.frame_encoder.parameters()) + list(self.head.parameters()),
                    self.learning_rate,
                    momentum=0.9,
                    weight_decay=0,
                    )
            elif self.distill_mode == "clip->frame":
                optimizer = torch.optim.SGD(
                    list(self.encoder.encoder.clip_encoder.parameters()) + list(self.head.parameters()),
                    self.learning_rate,
                    momentum=0.9,
                    weight_decay=0,
                    )
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.01, type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser

