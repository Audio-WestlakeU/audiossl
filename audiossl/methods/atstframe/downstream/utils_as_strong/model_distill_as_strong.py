'''
This file is modified from model_as_strong.py
The distillation model requires extra training losses, thus, we write a new model for it.
'''
import torch
import yaml
import os

import pandas as pd
import numpy as np
from torch import nn
from pytorch_lightning import LightningModule

from torch.nn import functional as F
from audiossl.datasets.as_strong_utils.as_strong_dict import get_lab_dict
from audiossl.datasets.dcase_utils import ManyHotEncoder
from audiossl.utils.common import cosine_scheduler_epoch
from audiossl.methods.atstframe.downstream.utils_psds_eval import evaluation, psds
from audiossl.methods.atstframe.downstream.utils_psds_eval.gpu_decode import (
    decode_preds,
    MedianPool2d,
    SEDMetrics,
    gpu_decode_preds
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, temp=1):
        # flatten
        x = x.transpose(1, 2)
        if self.use_norm:
            x = x.unsqueeze(-1)
            x = self.norm(x)
        x = x.squeeze(-1).transpose(1, 2)
        # linear layer + get strong predictions
        strong = self.sigmoid(self.linear(x) / temp)
        return strong.transpose(1, 2)

class DistillPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 max_epochs,
                 lr_scale,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 dcase_conf="./conf/frame_atst_as_strong.yaml",
                 multi_label=False,
                 metric_save_dir=None,
                 freeze_mode=False,
                 distill_mode="clip->frame"):
        super().__init__()
        self.distill_mode = distill_mode
        self.freeze_mode = freeze_mode
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch
        self.metric_save_dir = metric_save_dir
        self.encoder = encoder
        self.head = LinearHead(encoder.embed_dim, num_labels, use_norm=False, affine=False)
        self.multi_label = multi_label
        self.num_labels = num_labels
        self.loss_fn = torch.nn.BCELoss()
        self.monitor = 0
        self.val_loss = []
        self.save_hyperparameters(ignore=["encoder", ])
        with open(dcase_conf, "r") as f:
            self.config = yaml.safe_load(f)
        classes_labels = get_lab_dict(self.config["data"]["label_dict"])
        self.pred_decoder = ManyHotEncoder(
            list(classes_labels.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )

        # buffer for event based scores which we compute using sed-eval
        self.median_filter = MedianPool2d(7, same=True)
        self.sed_metrics_student = SEDMetrics(intersection_thd=0.5)

        test_n_thresholds = 50
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()
        self.lr_scale = lr_scale


    def training_step(self, batch, batch_idx):
        if not self.freeze_mode:
            self.encoder.finetune_mannual_train()
        self.schedule()
        data, labels, _ = batch

        x, labels = self.encoder((data, labels))
        if self.distill_mode == "clip->frame":
            with torch.no_grad():
                strong_pred_tea = self.encoder.encoder.teacher_module((data, labels))
            strong_pred_std = self.head(x)
            
        elif self.distill_mode == "frame->clip":
            with torch.no_grad():
                strong_pred_tea = self.encoder.encoder.teacher_module((data, labels))
            strong_pred_std= self.head(x)
        
        # Distillation loss
        loss_d_strong = self.loss_fn(strong_pred_std, strong_pred_tea.detach())
        strong_loss = self.loss_fn(strong_pred_std, labels)
        tot_loss = strong_loss / 2 + loss_d_strong / 2
        
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        self.log("train/synth/strong_loss", strong_loss)
        self.log("train/distill_loss", loss_d_strong)

        return tot_loss

    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            if type(self.mylr_scheduler) is list:
                param_group["lr"] = self.mylr_scheduler[i][self.global_step]
            else:
                param_group["lr"] = self.mylr_scheduler[self.global_step]
            # param_group["lr"] = self.learning_rate
        self.log("lr", param_group["lr"], prog_bar=True, logger=True)
   

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        data, labels, filenames = batch
        x, labels = self.encoder((data, labels))
        strong_pred = self.head(x)
        loss_strong_student = self.loss_fn(
            strong_pred, labels
        )
        self.val_loss.append(loss_strong_student.item())
        decoded_student_strong = decode_preds(strong_pred, [0.5], self.median_filter)
        self.sed_metrics_student.accm_macro_f1(decoded_student_strong, labels)

    def on_validation_epoch_end(self):
        # synth dataset
        intersection_f1_macro_student = self.sed_metrics_student.compute_macro_f1()
        val_loss = torch.tensor(np.mean(self.val_loss))
        obj_metric = torch.tensor(intersection_f1_macro_student.item())
        self.log("val/synth/student/loss_strong", val_loss)
        self.log("val/object_metric", val_loss, prog_bar=True)
        self.log("val/synth/student/intersection_f1_macro", intersection_f1_macro_student)
        return obj_metric
    
    
    def test_step(self, batch, batch_idx):
        self.encoder.eval()

        x, labels, filenames = batch
        x, labels = self.encoder((x, labels))

        strong_pred = self.head(x)
        # Get weak label for real data
        test_loss = self.loss_fn(strong_pred, labels)
        self.log("test/real/strong_loss", test_loss, prog_bar=True, logger=True)
        # Compute PSDS (Different from F1 metric, PSDS computes the ROC, which requires various thresholds from 0 to 1.)   
        decoded_strong = gpu_decode_preds(
            strong_pred, 
            thresholds=list(self.test_psds_buffer.keys()),
            filenames=filenames,
            encoder=self.pred_decoder, 
            median_filter=self.median_filter
        )
        for th in self.test_psds_buffer.keys():
            self.test_psds_buffer[th] = pd.concat([self.test_psds_buffer[th], decoded_strong[th]], ignore_index=True)
        # Compute F1 metric
        mid_val = list(self.test_psds_buffer.keys())[len(self.test_psds_buffer.keys()) // 2]
        self.decoded_05_buffer = self.decoded_05_buffer.append(decoded_strong[mid_val])
        return

    def on_test_epoch_end(self) -> None:
        save_dir = os.path.join(self.metric_save_dir, "metrics_test")
        # # calculate the metrics
        evaluation.g_parallel=True
        psds.g_parallel=True
        
        psds_score_scenario1 = compute_psds_from_operating_points(
            self.test_psds_buffer,
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=0.0,
            save_dir=os.path.join(save_dir, "scenario1"),
            weighted=False,
        )

        psds_score_scenario2 = compute_psds_from_operating_points(
            self.test_psds_buffer,
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=0.0,
            save_dir=os.path.join(save_dir, "scenario2"),
            weighted=False,
        )
        #==================================

        intersection_f1_macro = compute_per_intersection_macro_f1(
            {"0.5": self.decoded_05_buffer},
            self.config["data"]["test_tsv"],
            self.config["data"]["test_dur"],
        )
        print("Intersection F1:", intersection_f1_macro)
        best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2))

        results = {
            "hp_metric": best_test_result,
            "test/real/psds_score_scenario1": psds_score_scenario1,
            "test/real/psds_score_scenario2": psds_score_scenario2,
            # "test/real/event_f1_macro": event_F1_macro * 100,
            "test/real/intersection_f1_macro": intersection_f1_macro * 100,
        }
        print(results)
        if self.logger is not None:
                self.logger.log_metrics(results)
                self.logger.log_hyperparams(self.config, results)

        for key in results.keys():
            self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        if self.lr_scale == 1:
            optimizer = torch.optim.SGD(
            list(self.encoder.encoder.parameters()) + list(self.head.parameters()),
            self.learning_rate,
            momentum=0.9,
            weight_decay=0,
            )
            self.mylr_scheduler = cosine_scheduler_epoch(self.learning_rate, 1e-6, self.max_epochs, self.niter_per_epoch, self.warmup_epochs)
        else:
            param_groups, lower_lrs = self.request_param_groups()
            # # Linear layer
            param_groups.append({"params": self.head.parameters(), "lr": self.learning_rate})
            lower_lrs.append(1e-6)
            self.mylr_scheduler = [
                cosine_scheduler_epoch(x["lr"], lower_lrs[i], self.max_epochs, self.niter_per_epoch, self.warmup_epochs) for i, x in enumerate(param_groups)
            ]
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=0,
                )            
        return [optimizer]

    def request_param_groups(self):
        # Used for FrameATST only
        tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.encoder.encoder.named_parameters():
            if "teacher" in k:
                continue
            if "blocks.0." in k:
                tfm_params[1].append(p)
            elif "blocks.1." in k:
                tfm_params[2].append(p)
            elif "blocks.2." in k:
                tfm_params[3].append(p)
            elif "blocks.3." in k:
                tfm_params[4].append(p)
            elif "blocks.4." in k:
                tfm_params[5].append(p)
            elif "blocks.5." in k:
                tfm_params[6].append(p)
            elif "blocks.6." in k:
                tfm_params[7].append(p)
            elif "blocks.7." in k:
                tfm_params[8].append(p)
            elif "blocks.8" in k:
                tfm_params[9].append(p)
            elif "blocks.9." in k:
                tfm_params[10].append(p)
            elif "blocks.10." in k:
                tfm_params[11].append(p)
            elif "blocks.11." in k:
                tfm_params[12].append(p)
            elif "norm_frame." in k:
                tfm_params[13].append(p)
            else:
                tfm_params[0].append(p)
            print(k)
        tfm_params = list(reversed(tfm_params))
        tfm_groups = [{"params": tfm_params[i], "lr": self.learning_rate * (self.lr_scale ** i)} for i in range(len(tfm_params))]
        lower_lrs = [1e-6 * (self.lr_scale ** i) for i in range(len(tfm_params))]
        return tfm_groups, lower_lrs

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.01, type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser

