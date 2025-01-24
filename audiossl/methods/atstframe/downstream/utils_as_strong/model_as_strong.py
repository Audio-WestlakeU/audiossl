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
from audiossl.methods.atstframe.downstream.utils_ccom_eval.my_decode import write_results
from audiossl.utils.common import cosine_scheduler_epoch
from audiossl.methods.atstframe.downstream.utils_psds_eval import evaluation, psds
from audiossl.methods.atstframe.downstream.utils_psds_eval.gpu_decode import (
    onehot_decode_preds,
    MedianPool2d,
    SEDMetrics,
    gpu_decode_preds
)
from audiossl.methods.atstframe.downstream.utils_psds_eval.evaluation import (
    compute_per_intersection_macro_f1,
    compute_psds_from_operating_points
)


'''
This file is modified from the dcase baseline code
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
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x, temp=1):
        # flatten
        x = x.transpose(1, 2)
        if self.use_norm:
            x = x.unsqueeze(-1)
            x = self.norm(x)
        x = x.squeeze(-1).transpose(1, 2)
        # linear layer + get strong predictions
        strong = self.linear(x) / temp
        return strong.transpose(1, 2)


class MLPHead(nn.Module):
    def __init__(self, feature_dim, num_labels, frame=True) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = 512
        self.proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_dim, num_labels)
        self.frame = frame

    def forward(self, x):
        # 输入形状[64, 250, 768]
        x = self.proj(x)
        if not self.frame:
            x = x.mean(1, False)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x.transpose(1, 2)

class FineTuningPLModule(LightningModule):
    def __init__(self,
                 encoder,
                 learning_rate,
                 max_epochs,
                 niter_per_epoch,
                 warmup_epochs,
                 num_labels,
                 dcase_conf="./conf/frame_atst_as_strong.yaml",
                 multi_label=False,
                 metric_save_dir=None,
                 freeze_mode=False,
                 lr_scale=1.0,
                 loss_weights=None,
                 classifier='linear'):
        super().__init__()
        self.freeze_mode = freeze_mode
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.niter_per_epoch = niter_per_epoch
        self.metric_save_dir = metric_save_dir
        self.encoder = encoder

        if classifier == 'linear':
            self.head = LinearHead(encoder.embed_dim, num_labels, use_norm=False, affine=False)
        elif classifier == 'mlp':
            self.head = MLPHead(encoder.embed_dim, num_labels)
        else:
            raise NotImplementedError(f"{classifier} classifier defined in yaml is not supported. Double check it.")
        self.multi_label = multi_label
        self.num_labels = num_labels

        self.monitor = 0
        self.val_loss = []
        self.save_hyperparameters(ignore=["encoder", ])
        with open(dcase_conf, "r") as f:
            self.config = yaml.safe_load(f)
        self.classes_labels = get_lab_dict(self.config["data"]["label_dict"])
        self.pred_decoder = ManyHotEncoder(
            list(self.classes_labels.keys()),
            audio_len=self.config["data"]["audio_max_len"],
            frame_len=self.config["feats"]["n_filters"],
            frame_hop=self.config["feats"]["hop_length"],
            net_pooling=self.config["data"]["net_subsample"],
            fs=self.config["data"]["fs"],
        )
        # 使用一个bool来开启CrossEntropyLoss，与MERT等finetune对齐
        if  self.config["train"]["loss"] == 'ce':
            if self.multi_label:
                raise Exception('同一个frame多标签的情况下，不能使用CrossEntropyLoss. 这是在datasets.__init__里定义的')
            self.use_ce_loss = True
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
            print("Using CrossEntropyLoss with weights: ", loss_weights)
            self.test_psds_buffer = {0: pd.DataFrame()}  # 只用一个0作为替代
            self.test_results = {'filenames': [], 'predictions': []} # 存储所有predictions
        else:
            self.use_ce_loss = False
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0, 2.0, 10.0, 5.0, 2.0, 1.5, 1.5, 1.0]))
            print("Using BCEWithLogitsLoss...")
            test_n_thresholds = 50
            test_thresholds = np.arange(
                1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
            )
            self.test_psds_buffer = {k: pd.DataFrame() for k in test_thresholds}
            self.decoded_05_buffer = pd.DataFrame()
            self.decoded_025_buffer = pd.DataFrame()

        # buffer for event based scores which we compute using sed-eval
        self.median_filter = MedianPool2d(7, same=True)
        self.sed_metrics_student = SEDMetrics(intersection_thd=0.5)
        self.validation_outputs = []

        self.lr_scale = lr_scale

    def training_step(self, batch, batch_idx):        
        if self.freeze_mode:
            self.encoder.eval()
        else:
            self.encoder.finetune_mannual_train()
        self.schedule()
        x, labels, _ = batch
        x, labels = self.encoder((x, labels))

        strong_pred = self.head(x)  # Linear head contains transpose[1,2] #[batch, time, class_num]

        # Get weak label for real data
        strong_loss = self._calculate_loss(strong_pred, labels)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True)
        self.log("train/strong_loss", strong_loss)
        return strong_loss

    def _calculate_loss(self, strong_pred, labels):
        # 输入的strong_pred, labels: [bsz, cls, T][64, 9, 250]
        # 这里把pred和labels统一成(bsz, T, cls), 默认类别是最后一个维度，就可以使用pos_weight=tensor(num_classes)
        strong_pred, labels = strong_pred.transpose(1,2), labels.transpose(1, 2) # [bsz, T, cls][64, 250, 9]
        if not self.use_ce_loss:   # 源代码默认使用bce loss
            return self.loss_fn(strong_pred, labels)

        # 以下是要用CrossEntropyLoss需要做的改动
        # 转化shape: (64, 250, 9) ---> (64, 250) 在ASStrongDataset中使用的是ManyHotEncoder, init确认过了不是multilabel
        targets_single_label = labels.argmax(dim=-1)  # (64, 250)，在label_dict里，所有的label都是1～8，当都是0的时候，返回的就是第一个index正好对应0无标签
        # 展平 logits(strong_pred) 和 targets (labels)
        _B, _T, C = strong_pred.shape
        assert C == self.num_labels
        logits_flat = strong_pred.view(-1, C)  # shape: (64 * 250, 9)
        targets_flat = targets_single_label.view(-1)  # shape: (64 * 250,)

        return self.loss_fn(logits_flat, targets_flat)  # return CrossEntropyLoss

    def _calculate_framewise_metrics(self):
        from sklearn.metrics import f1_score, accuracy_score, classification_report
        # 将所有 batch 的预测和真实标签拼接起来
        strong_pred = torch.cat([x['preds'] for x in self.validation_outputs], dim=0)  # 形状: [sample_size, cls, T]
        labels = torch.cat([x['labels'] for x in self.validation_outputs], dim=0)  # 形状: [sample_size, cls, T]
        strong_pred, labels = strong_pred.transpose(1, 2), labels.transpose(1, 2)  # [sample_size, T, cls][64, 250, 9]

        # 对 strong_pred 进行 Softmax 归一化
        probs = F.softmax(strong_pred, dim=-1)
        # 进行 argmax，获取预测类别
        preds = torch.argmax(probs, dim=-1)  # shape: [sample_size, T]
        labels = torch.argmax(labels, dim=-1)  # shape: [sample_size, T]
        # 展开 preds 和 labels
        preds_flat = preds.view(-1)  # shape: [sample_size * T]
        labels_flat = labels.view(-1)  # shape: [sample_size * T]

        # 将张量转换为 NumPy 数组
        preds_flat_np = preds_flat.cpu().numpy()  # 如果张量在 GPU 上，需要先移动到 CPU
        labels_flat_np = labels_flat.cpu().numpy()

        # 计算 F1 Score
        f1 = f1_score(labels_flat_np, preds_flat_np, average='macro')  # 使用 'macro' 平均
        accuracy = accuracy_score(labels_flat_np, preds_flat_np)

        # 使用 classification_report 打印每个类别的详细指标
        unique_labels = np.unique(np.concatenate([labels_flat_np, preds_flat_np]))
        report = classification_report(labels_flat_np, preds_flat_np, labels=unique_labels)

        return f1, accuracy, report

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
        x, labels, filenames = batch
        x, labels = self.encoder((x, labels))

        strong_pred = self.head(x)  # [bsz, cls, T] [64, 9, 250]
        # 原代码计算sed_metrics
        decoded_student_strong = onehot_decode_preds(strong_pred, [0], self.median_filter) # [bsz, cls, T]
        self.sed_metrics_student.accm_macro_f1(decoded_student_strong, labels)
        # 现在计算framewise
        self.validation_outputs.append({
            'preds': strong_pred,  # 预测概率
            'labels': labels  # 真实标签
        })
        loss_strong_student = self._calculate_loss(strong_pred, labels)
        self.val_loss.append(loss_strong_student.item())

    def on_validation_epoch_end(self):
        intersection_f1_macro_student = self.sed_metrics_student.compute_macro_f1()
        val_loss = torch.tensor(np.mean(self.val_loss))
        f1_score, accuracy, report = self._calculate_framewise_metrics()
        print(f'on_validation_epoch_end, epoch {self.current_epoch} classification_report')
        print(report)
        print(f'f1_score: {f1_score}', f'accuracy: {accuracy}')
        self.log("val/loss", val_loss)
        self.log("val/f1_score", f1_score)
        self.log("val/accuracy", accuracy)
        self.log("val/student/intersection_f1_macro", intersection_f1_macro_student)

    def test_step(self, batch, batch_idx):
        self.encoder.eval()

        x, labels, filenames = batch
        x, labels = self.encoder((x, labels))

        strong_pred = self.head(x)  # [bsz, cls, T] [64, 9, 250]

        # 计算loss
        test_loss = self._calculate_loss(strong_pred, labels)
        self.log("test/real/strong_loss", test_loss, prog_bar=True, logger=True)

        # Compute PSDS (Different from F1 metric, PSDS computes the ROC, which requires various thresholds from 0 to 1.)
        # 在每一步test_step不能直接decode，原因是要对截成10s的片段进行合并结果，这一步只能先保存，最后test_epoch_end再合并+decode
        self.test_results['filenames'].extend(filenames)
        self.test_results['predictions'].append(strong_pred)

        # decoded_strong = onehot_gpu_decode_preds(
        #     strong_pred,
        #     thresholds=list(self.test_psds_buffer.keys()),
        #     filenames=filenames,
        #     encoder=self.pred_decoder,
        #     median_filter=self.median_filter
        # )
        # for th in self.test_psds_buffer.keys():
        #     self.test_psds_buffer[th] = pd.concat([self.test_psds_buffer[th], decoded_strong[th]], ignore_index=True)
        # # Compute F1 metric
        # if not self.use_ce_loss:
        #     mid_val = list(self.test_psds_buffer.keys())[len(self.test_psds_buffer.keys()) // 2]
        #     self.decoded_05_buffer = pd.concat([self.decoded_05_buffer, decoded_strong[mid_val]], ignore_index=True)
        #     # use other values instead of mid_val
        #     quarter_val = list(self.test_psds_buffer.keys())[len(self.test_psds_buffer.keys()) // 4]
        #     self.decoded_025_buffer = pd.concat([self.decoded_025_buffer, decoded_strong[quarter_val]],ignore_index=True)
        # return

    def on_test_epoch_end(self) -> None:
        # 只保存预测结果csv，评价指标的计算统一到utils_ccom_eval/evaluation.py，与其他方法一起做对比
        save_dir = os.path.join(self.metric_save_dir, "metrics_test")
        os.makedirs(save_dir, exist_ok=True)
        # calculate the metrics
        # Enable parallel computing
        evaluation.g_parallel=True
        psds.g_parallel=True

        # save self.decoded_buffer as tsv file, remove NA
        if not self.use_ce_loss:
            self.decoded_05_buffer.to_csv(os.path.join(save_dir, "pred_0.5.csv"), index=False)
            self.decoded_025_buffer.to_csv(os.path.join(save_dir, "pred_0.25.csv"), index=False)
            print(f"Saved pred_0.5.csv, pred_0.25.csv to {save_dir}.")

            # intersection_f1_macro, f_dict = compute_per_intersection_macro_f1(
            #     {"0.5": self.decoded_05_buffer},
            #     self.config["data"]["test_tsv"],
            #     self.config["data"]["test_dur"],
            # )
        else:
            write_results(filenames=self.test_results['filenames'], predictions=self.test_results['predictions'],
                          save_dir=save_dir)

            # merge_preds = torch.cat(self.test_results["predictions"], dim=0)  # 在第一个batch的维度拼接起来
            # sample_size, C, T = merge_preds.shape
            # assert sample_size == len(self.test_results["filenames"])
            # print('predictions after merge: ', merge_preds.shape)
            # print(self.test_results["filenames"])
            #
            # assert list(self.test_psds_buffer.keys()) == [0]
            # df = self.test_psds_buffer[0]
            # df_cleaned = df[df['event_label'] != 'NA']
            # df_cleaned.to_csv(os.path.join(save_dir, "pred_0.csv"), index=False)
            # print(f"Saved pred_0.csv to {save_dir}.")

            # intersection_f1_macro, f_dict = compute_per_intersection_macro_f1(
            #     {"0": df_cleaned},
            #     self.config["data"]["test_tsv"],
            #     self.config["data"]["test_dur"],
            # )

        # print("Intersection F1:", intersection_f1_macro)
        # print("Intersection F1 per category:", f_dict)
        #
        # psds_score_scenario1 = compute_psds_from_operating_points(
        #     self.test_psds_buffer,
        #     self.config["data"]["test_tsv"],
        #     self.config["data"]["test_dur"],
        #     dtc_threshold=0.7,
        #     gtc_threshold=0.7,
        #     alpha_ct=0,
        #     alpha_st=0.0,
        #     save_dir=os.path.join(save_dir, "scenario1"),
        #     weighted=False,
        # )
        # psds_score_scenario2 = compute_psds_from_operating_points(
        #     self.test_psds_buffer,
        #     self.config["data"]["test_tsv"],
        #     self.config["data"]["test_dur"],
        #     dtc_threshold=0.1,
        #     gtc_threshold=0.1,
        #     cttc_threshold=0.3,
        #     alpha_ct=0.5,
        #     alpha_st=0.0,
        #     save_dir=os.path.join(save_dir, "scenario2"),
        #     weighted=False,
        # )
        #
        #
        # best_test_result = torch.tensor(max(psds_score_scenario1, psds_score_scenario2))
        #
        # results = {
        #     "hp_metric": best_test_result,
        #     "test/real/psds_score_scenario1": psds_score_scenario1,
        #     "test/real/psds_score_scenario2": psds_score_scenario2,
        #     # "test/real/event_f1_macro": event_F1_macro * 100, # omit computing EBF1, takes to much time!
        #     "test/real/intersection_f1_macro": intersection_f1_macro * 100,
        # }
        # print(results)
        # if self.logger is not None:
        #         self.logger.log_metrics(results)
        #         self.logger.log_hyperparams(self.config, results)
        #
        # for key in results.keys():
        #     self.log(key, results[key], prog_bar=True, logger=False)

    def configure_optimizers(self):
        # Freezing opt
        if self.freeze_mode:
            optimizer = torch.optim.SGD(
                list(self.head.parameters()),
                self.learning_rate,
                momentum=0.9,
                weight_decay=0,
                )
            self.mylr_scheduler = cosine_scheduler_epoch(self.learning_rate, 1e-6, self.max_epochs, self.niter_per_epoch, self.warmup_epochs)
        
        # Finetune opt
        else:
            if self.lr_scale == 1:
                optimizer = torch.optim.SGD(
                list(self.encoder.encoder.parameters()) + list(self.head.parameters()),
                self.learning_rate,
                momentum=0.9,
                weight_decay=0,
                )
                self.mylr_scheduler = cosine_scheduler_epoch(self.learning_rate, 1e-6, self.max_epochs, self.niter_per_epoch, self.warmup_epochs)
            else:
                print("Using scaling learning rate for ATST models")
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

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = parent_parser.add_argument_group("FineTuningModel")
        parser.add_argument("--learning_rate", default=0.01, type=float, help="""Learning rate""")
        parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--warmup_epochs', default=5, type=int)
        return parent_parser

    def request_param_groups(self):
        # Used for FrameATST only
        tfm_params = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.encoder.encoder.named_parameters():
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
        tfm_params = list(reversed(tfm_params))
        tfm_groups = [{"params": tfm_params[i], "lr": self.learning_rate * (self.lr_scale ** i)} for i in range(len(tfm_params))]
        lower_lrs = [1e-6 * (self.lr_scale ** i) for i in range(len(tfm_params))]
        print("layer-wise learning rate:", [ "{:.3f}".format(x["lr"]) for x in tfm_groups])
        return tfm_groups, lower_lrs