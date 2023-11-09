import torch
import yaml

import pandas as pd
import numpy as np
import torchmetrics as tm

from torch import nn
from pytorch_lightning import LightningModule

from torch.nn import functional as F

from class_dict import classes_labels
from audiossl.models.atst import audio_transformer
from audiossl.datasets.dcase_utils import ManyHotEncoder
from audiossl.utils.common import cosine_scheduler_epoch
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

from audiossl.methods.atstframe.downstream.comparison_models.clip_atst_module import ATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.frame_atst_module import FrameATSTPredModule
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


class FineTuningPLModule(LightningModule):
    def __init__(self,
                 mode="clip",
                 learning_rate=1e-3,
                 dcase_conf="./utils_dcase/conf/dcase_dataset.yaml",
                 max_epochs=100,
                 niter_per_epoch=20,
                 warmup_epochs=10,
                 num_labels=10,
                 n_last_blocks=1,
                 multi_label=False,
                 mixup_training=False,
                 metric_save_dir=None,
                 freeze_mode=False):
        super().__init__()
        self.freeze_mode = freeze_mode
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warumup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch
        self.metric_save_dir = metric_save_dir
        if mode == "clip":
            pretrained_module = ATSTPredModule("./comparison_models/ckpts/clip_atst.ckpt")
        elif mode == "frame":
            pretrained_module = FrameATSTPredModule("./comparison_models/ckpts/frame_atst.ckpt")
        self.encoder = pretrained_module
        self.head = LinearHead(768, 10, use_norm=False, affine=False)
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

        # buffer for event based scores which we compute using sed-eval
        self.median_filter = MedianPool2d(self.config["training"]["median_window"], same=True)
        self.sed_metrics_student = SEDMetrics(intersection_thd=0.5)

        
        test_n_thresholds = self.config["training"]["n_test_thresholds"]
        test_thresholds = np.arange(
            1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
        )
        self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
        self.decoded_05_buffer = pd.DataFrame()

    def forward(self, batch):
        self.encoder.eval()
        x, labels = self.encoder(batch)
        strong_pred, weak_pred = self.head(x)
        return strong_pred, weak_pred