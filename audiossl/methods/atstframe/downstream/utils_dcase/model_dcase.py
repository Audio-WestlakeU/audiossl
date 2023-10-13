import chunk
import torch
import yaml
import os

import pandas as pd
import numpy as np

from pathlib import Path
from torch import nn
from torchmetrics import F1Score
from pytorch_lightning import LightningModule

from torch.nn import functional as F

from audiossl.methods.pyramid import audio_transformer
from audiossl.methods.pyramid.downstream.utils import Metric
from audiossl.datasets.dcase_utils import ManyHotEncoder
from audiossl.utils.common import cosine_scheduler_epoch, get_params_groups
from .evaluation_measures import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points,
    batched_decode_preds,
    log_sedeval_metrics,
    classes_labels
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

    def forward(self, x):
        # flatten
        x = x.transpose(1, 2)
        if self.use_norm:
            x = x.unsqueeze(-1)
            x = self.norm(x)
        x = x.squeeze(-1).transpose(1, 2)
        # linear layer + get strong predictions
        strong = self.sigmoid(self.linear(x))

        # linear layer + get weak predictions
        soft = self.softmax(self.linear_softmax(x)).clamp(min=1e-7, max=1)
        weak = (strong * soft).sum(1) / soft.sum(1)
        return strong.transpose(1, 2), weak


class PretrainedEncoderPLModule(LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self,
                 pretrained_encoder: audio_transformer.FrameAST,
                 n_blocks: int,
                 avgpool: bool = True):
        super().__init__()
        self.encoder = pretrained_encoder
        self.n_blocks = n_blocks
        self.embed_dim = self.encoder.embed_dim*n_blocks

    def forward(self, batch):
        (x, length), y = batch
        x = x.unsqueeze(1)
        x = self.encoder.get_intermediate_layers(
            x,
            length,
            self.n_blocks,
            scene=False
        )
        # [MARK] We get the first element which is the output of the last layer pyramid. Temporary usage only.
        # Also drop the CLS token in this clip-level pyramid model
        # Transpose x to get in shape [bsz, 768 (h_dim), 250 (t_frame)]
        return x, y

class FineTuningPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 dcase_conf,
                 max_epochs,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 multi_label=False,
                 mixup_training=False,
                 metric_save_dir=None,
                 **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
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

        self.save_hyperparameters(ignore=["encoder"])
        with open(dcase_conf, "r") as f:
            self.config = yaml.safe_load(f)

        # Metrics for DCASE validation and test
        self.val_f1 = F1Score(
            num_labels,
            average="macro",
            compute_on_step=False,
        )

        self.val_buffer_synth = {k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]}
        self.test_buffer_synth = {k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]}

        self.pred_decoder = ManyHotEncoder(
            list(classes_labels.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )

        test_n_thresholds = self.config["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()


    def training_step(self, batch, batch_idx):
        bsz = self.config["training"]["batch_size"]
        indx_synth, indx_weak = bsz

        self.encoder.train()
        self.schedule()
        x, labels, _ = batch
        x, labels = self.encoder((x, labels))
        # Get prediction [bsz, cls, time]
        strong_pred, weak_pred = self.head(x)

        # Get weak and strong mask for two types of data
        strong_mask = torch.zeros(indx_synth + indx_weak).to(x).bool()
        weak_mask = torch.zeros(indx_synth + indx_weak).to(x).bool()
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth : indx_weak + indx_synth] = 1

        # Get weak label for real data
        labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
        weak_loss = self.loss_fn(weak_pred[weak_mask], labels_weak)
        strong_loss = self.loss_fn(strong_pred[strong_mask], labels[strong_mask])

        tot_loss = weak_loss + strong_loss
        
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        self.log("train/real/weak_loss", weak_loss)
        self.log("train/synth/strong_loss", strong_loss)
        self.log("train/total_loss", tot_loss)
        self.optimizers()
        return tot_loss

    def schedule(self):
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            # param_group["lr"] = self.mylr_scheduler[self.global_step]
            param_group["lr"] = self.learning_rate
        self.log("lr", param_group["lr"], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        x, labels, filenames = batch
        x, labels = self.encoder((x, labels))
        strong_pred, weak_pred = self.head(x)
        # Get weak and strong mask for two types of data
        weak_mask = torch.tensor([str(Path(f).parent) == str(Path(self.config["data"]["weak_folder"])) for f in filenames]).to(x).bool()
        strong_mask = torch.tensor([str(Path(f).parent) == str(Path(self.config["data"]["synth_val_folder"])) for f in filenames]).to(x).bool()

        if torch.any(weak_mask):
            # Get weak label for real data
            labels_weak = (torch.sum(labels[weak_mask], -1) > 0).float()
            # Validation weak loss
            weak_loss = self.loss_fn(weak_pred[weak_mask], labels_weak)
            # Validation weak macro cumulator
            self.val_f1(weak_pred[weak_mask], labels_weak.int())
            self.log("val/real/weak_loss", weak_loss, prog_bar=False, logger=True)
    
        if torch.any(strong_mask):
            # Validation strong loss
            strong_loss = self.loss_fn(strong_pred[strong_mask], labels[strong_mask])
            self.log("val/synth/strong_loss", strong_loss, prog_bar=False, logger=True)
            # Restore predictions for strong predictions
            filenames_synth = [x for x in filenames if Path(x).parent == Path(self.config["data"]["synth_val_folder"])]
            decoded_strong = batched_decode_preds(
                strong_pred[strong_mask],
                filenames_synth,
                self.pred_decoder,
                median_filter=self.config["training"]["median_window"],
                thresholds=list(self.val_buffer_synth.keys()),
            )
            for th in self.val_buffer_synth.keys():
                self.val_buffer_synth[th] = pd.concat([self.val_buffer_synth[th], decoded_strong[th]], ignore_index=True)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        return

    def on_validation_epoch_end(self):
        # Weak metric compute
        weak_f1_macro = self.val_f1.compute()
        # Strong metric compute
        intersection_f1_macro = compute_per_intersection_macro_f1(
            self.val_buffer_synth,
            self.config["data"]["synth_val_tsv"],
            self.config["data"]["synth_val_dur"],
        )
        event_f1_macro = log_sedeval_metrics(
            self.val_buffer_synth[0.5], 
            self.config["data"]["synth_val_tsv"]
        )[0]
        # Choose best intersection performance [MARK]
        obj_metric = intersection_f1_macro
        obj_metric = torch.tensor(weak_f1_macro.item() + obj_metric)

        # Monitoring and logging
        self.monitor = obj_metric
        self.log("val/object_metric", self.monitor, prog_bar=True, logger=True)
        self.log("val/real/macro_F1", weak_f1_macro)
        self.log("val/synth/intersection_macro_F1", intersection_f1_macro)
        self.log("val/synth/event_macro_F1", event_f1_macro)

         # Free the buffers
        self.val_buffer_synth = {k: pd.DataFrame() for k in self.config["training"]["val_thresholds"]}
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        self.encoder.eval()

        x, labels, filenames = batch
        x, labels = self.encoder((x, labels))
        strong_pred, _ = self.head(x)

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
        optimizer = torch.optim.SGD(list(self.encoder.encoder.parameters())+list(self.head.parameters()),
                                    self.learning_rate,
                                    momentum=0.9,
                                    weight_decay=0,
                                    )
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.002, type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser

