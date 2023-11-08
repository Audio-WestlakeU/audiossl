import os
import scipy
import math
import json
import torch

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from .evaluation import compute_sed_eval_metrics
from scipy import stats
from pathlib import Path

from sklearn.metrics import auc
from torch.nn.modules.utils import _quadruple

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.shape[-1]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - (iw % self.stride), 0)
            pl = pw // 2
            pr = pw - pl
            padding = (pr, pl, 0, 0)
        else:
            padding = self.padding
        return padding

    def median(self, x, dim, keepdim=False):
        """
        Find the median along a particular dimension.

        If the dimension length is even take the average of the central values.

        Use *keepdim=True* to preserve the dimension after reduction.
        """
        index = torch.argsort(x, dim)
        deref = [slice(None, None)]*len(x.shape)
        middle = x.shape[dim]//2
        even = 1 - x.shape[dim]%2
        deref[dim] = slice(middle-even, middle+1+even)
        values = x.gather(dim, index[deref])
        return (
            values.mean(dim, keepdim=keepdim) if even 
            else values if keepdim 
            else values.squeeze(dim))

    def scripy_pad(self, x, padding):
        assert len(x.shape) == 4, "wrong x shape"
        x = F.pad(x, (1, 1, 0, 0), mode='constant')
        x = F.pad(x, padding, mode="reflect")
        x = torch.cat([
            x[:, :, :, :padding[0]],
            x[:, :, :, padding[0] + 1: -padding[1] - 1],
            x[:, :, :, -padding[1]:],
        ], dim=-1)
        return x

    def forward(self, x):
        with torch.no_grad():
            if len(x.shape) == 3:
                single = True
                x = x.unsqueeze(0)
            else:
                single = False
            x = self.scripy_pad(x, self._padding(x))
            x = x.unfold(3, self.k, self.stride)
            x = self.median(x, dim=-1)
            if single:
                x = x.squeeze(0)
        return x

class SEDMetrics(nn.Module):
    def __init__(self, intersection_thd=0.7):
        super(SEDMetrics, self).__init__()
        self.intersection_thd = intersection_thd
        self.reset_stats()
    
    def reset_stats(self):
        self.tps = 0
        self.fps = 0
        self.fns = 0
        self.tns = 0

    def compute_truth_table(self, strong_preds, ground_truth):
        with torch.no_grad():
            
            bsz, num_cls, T = strong_preds.shape
            preds = strong_preds.bool()
            labels = ground_truth.bool()

            idv_event_triu = torch.cuda.FloatTensor(T + 1, T).fill_(1).triu().T

            # Locate each events
            all_events = torch.logical_or(preds, labels).float()
            events_bdry = torch.cat([all_events, torch.cuda.FloatTensor(bsz, num_cls, 1).fill_(0)], dim=-1) \
                - torch.cat([torch.cuda.FloatTensor(bsz, num_cls, 1).fill_(0), all_events], dim=-1)
            events_start = torch.argwhere(events_bdry == 1)
            events_end = torch.argwhere(events_bdry == -1)
        
            # Get individual events out from the pred, label, and OR(pred, label)
            pred_full_events = strong_preds[events_start[:, 0], events_start[:, 1], :]
            label_full_events = ground_truth[events_start[:, 0], events_start[:, 1], :]
            idv_event_mask = (torch.index_select(idv_event_triu, dim=1, index=events_start[:, -1]) - torch.index_select(idv_event_triu, dim=1, index=events_end[:, -1])).T

            # Get the cls one-hot according to events
            tp_compute = (pred_full_events * idv_event_mask).sum(-1) / ((label_full_events * idv_event_mask).sum(-1) + 1e-7)
            longer_preds = tp_compute >= self.intersection_thd
            shorter_preds = tp_compute < 1 / self.intersection_thd
            tp_full = torch.logical_and(longer_preds, shorter_preds)
            fp_full = torch.logical_xor(longer_preds, tp_full).float()
            fn_full = torch.logical_xor(shorter_preds, tp_full).float()
            tp_full = tp_full.float()
            return tp_full, fp_full, fn_full, events_start
            
    def compute_tn(self, strong_preds, neg_truths):
        # Similar to truth_table, here the neg_truths is used to compute tns
        # TNs: both pred and neg_truths are true for each frame
        with torch.no_grad():
            
            bsz, num_cls, T = strong_preds.shape
            idv_event_triu = torch.cuda.FloatTensor(T + 1, T).fill_(1).triu().T

            # Locate each events
            events_bdry = torch.cat([neg_truths, torch.cuda.FloatTensor(bsz, num_cls, 1).fill_(0)], dim=-1) \
                - torch.cat([torch.cuda.FloatTensor(bsz, num_cls, 1).fill_(0), neg_truths], dim=-1)
            events_start = torch.argwhere(events_bdry == 1)
            events_end = torch.argwhere(events_bdry == -1)
        
            # Get individual events out from the pred, label, and OR(pred, label)
            pred_full_events = strong_preds[events_start[:, 0], events_start[:, 1], :]
            idv_event_mask = (torch.index_select(idv_event_triu, dim=1, index=events_start[:, -1]) - torch.index_select(idv_event_triu, dim=1, index=events_end[:, -1])).T

            # Get the cls one-hot according to events
            tn_compute = (pred_full_events * idv_event_mask).sum(-1) / idv_event_mask.sum(-1)
            tn_full = (tn_compute == 1).float()
            return tn_full, events_start

    def compute_avg_f1(self, strong_preds, ground_truths):
        bsz, _, _ = strong_preds.shape
        event_eye = torch.eye(bsz, device=strong_preds.device)
        tps, _, _, events_index = self.compute_truth_table(strong_preds, ground_truths)
        event_to_clip = torch.index_select(event_eye, dim=0, index=events_index[:, 0])
        tp_clip = tps.unsqueeze(0).matmul(event_to_clip)
        tp_fn_fp_clip = event_to_clip.sum(0)
        # F1 score accoding to clips
        f_score = tp_clip / (1 / 2 * tp_clip + 1 / 2 * tp_fn_fp_clip)
        f_score = f_score.nan_to_num(0)
        return f_score.mean()

    def accm_macro_f1(self, strong_preds, ground_truths):
        _, num_cls, _ = strong_preds.shape
        cls_eye = torch.eye(num_cls, device=strong_preds.device)
        tp_full, fp_full, fn_full, events_index = self.compute_truth_table(strong_preds, ground_truths)
        cls_one_hot = torch.index_select(cls_eye, dim=0, index=events_index[:, 1])
        cls_tp = tp_full.unsqueeze(0).matmul(cls_one_hot)
        cls_fp = fp_full.unsqueeze(0).matmul(cls_one_hot)
        cls_fn = fn_full.unsqueeze(0).matmul(cls_one_hot)
        self.tps += cls_tp
        self.fps += cls_fp
        self.fns += cls_fn
    
    def compute_macro_f1(self):
        false_num = self.fps + self.fns
        if false_num is 0:
            false_num += torch.cuda.FloatTensor(1).fill_(1e-7)
        f_score = self.tps / (self.tps + 1 / 2 * (false_num))
        f_score = f_score.nan_to_num(0)
        self.reset_stats()
        return f_score.mean()

    def accm_auc(self, strong_preds, pos_truths, neg_truths):
        # strong_preds: [thds, bsz, cls, T]
        num_thds, _, num_cls, _ = strong_preds.shape
        # redefine tps with thds dimension
        self.tps += torch.cuda.FloatTensor(num_thds, num_cls).fill_(0)
        self.fps += torch.cuda.FloatTensor(num_thds, num_cls).fill_(0)
        self.fns += torch.cuda.FloatTensor(num_thds, num_cls).fill_(0)
        self.tns += torch.cuda.FloatTensor(num_thds, num_cls).fill_(0)
        # cls eye for one-hot calculation
        cls_eye = torch.eye(num_cls, device=strong_preds.device)
        for i, strong_preds_thd in enumerate(strong_preds):
            tp_full, fp_full, fn_full, events_index = self.compute_truth_table(strong_preds_thd, pos_truths)
            tn_full, neg_index = self.compute_tn(1 - strong_preds_thd, neg_truths)
            cls_one_hot = torch.index_select(cls_eye, dim=0, index=events_index[:, 1])
            neg_cls_one_hot = torch.index_select(cls_eye, dim=0, index=neg_index[:, 1])
            cls_tp = tp_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_fp = fp_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_fn = fn_full.unsqueeze(0).matmul(cls_one_hot).squeeze(0)
            cls_tn = tn_full.unsqueeze(0).matmul(neg_cls_one_hot).squeeze(0)

            self.tps[i] += cls_tp
            self.fps[i] += cls_fp
            self.fns[i] += cls_fn
            self.tns[i] += cls_tn

    def compute_auc(self):
        # F1 score according to labels
        tpr = self.tps / (self.tps + self.fps)
        # Calculate fpr by compute the TNs (this requires the explicit negative labels been given)
        fpr = self.fps / (self.fps + self.tns)
        cls_auc = []
        for i in range(tpr.shape[1]):
            fpr_np = fpr[:, i].cpu().numpy()
            tpr_np = tpr[:, i].cpu().numpy()
            cls_auc.append(auc(fpr_np[-1::-1], tpr_np[-1::-1]))
        auc_score = sum(cls_auc) / len(cls_auc)
        self.cls_auc = []
        self.reset_stats()
        return auc_score

    def compute_d_prime(self, auc):
        standard_normal = stats.norm()
        d_prime = standard_normal.ppf(auc) * math.sqrt(2.0)
        return d_prime

def decode_preds(strong_preds, thds, median_filter):
    bsz, cls, T = strong_preds.shape
    # Not implement for multiple thds
    if len(thds) > 1:
        smooth_preds = torch.cuda.FloatTensor(len(thds), bsz, cls, T)
    for i, thd in enumerate(thds):
        thd = torch.tensor(thd, device="cuda").reshape(1, 1, 1)
        thd = thd.repeat(bsz, cls, T)
        hard_preds = strong_preds > thd
        if len(thds) > 1:
            smooth_preds[i] = median_filter(hard_preds.float())
        else:
            smooth_preds = median_filter(hard_preds.float())

    return smooth_preds

def batched_decode_preds(
    strong_preds, filenames, encoder, thresholds=[0.5], median_filter=7, pad_indx=None,
):
    """ Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()
    strong_preds = strong_preds.detach().cpu().numpy()
    for j in range(strong_preds.shape[0]):  # over batches
        for k, c_th in enumerate(thresholds):
            c_preds = strong_preds[j]
            if pad_indx is not None:
                true_len = int(c_preds.shape[-1] * pad_indx[j].item())
                c_preds = c_preds[:true_len]
            pred = c_preds.T
            pred = pred > c_th
            pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            prediction_dfs[c_th] = pd.concat([prediction_dfs[c_th], pred], ignore_index=True)
    return prediction_dfs

def convert_to_event_based(weak_dataframe):
    """ Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
    ('filename', 'onset', 'offset', 'event_label').

    Args:
        weak_dataframe: pd.DataFrame, the dataframe to be converted.

    Returns:
        pd.DataFrame, the dataframe strongly labeled.
    """

    new = []
    for i, r in weak_dataframe.iterrows():

        events = r["event_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1}
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir=None, label_interest=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")
    if label_interest is not None:
        gt_mask = [x in label_interest for x in gt["event_label"]]
        gt = gt[gt_mask]
    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))
    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


def parse_jams(jams_list, encoder, out_json):

    if len(jams_list) == 0:
        raise IndexError("jams list is empty ! Wrong path ?")

    backgrounds = []
    sources = []
    for jamfile in jams_list:

        with open(jamfile, "r") as f:
            jdata = json.load(f)

        # check if we have annotations for each source in scaper
        assert len(jdata["annotations"][0]["data"]) == len(
            jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"]
        )

        for indx, sound in enumerate(jdata["annotations"][0]["data"]):
            source_name = Path(
                jdata["annotations"][-1]["sandbox"]["scaper"][
                    "isolated_events_audio_path"
                ][indx]
            ).stem
            source_file = os.path.join(
                Path(jamfile).parent,
                Path(jamfile).stem + "_events",
                source_name + ".wav",
            )

            if sound["value"]["role"] == "background":
                backgrounds.append(source_file)
            else:  # it is an event
                if (
                    sound["value"]["label"] not in encoder.labels
                ):  # correct different labels
                    if sound["value"]["label"].startswith("Frying"):
                        sound["value"]["label"] = "Frying"
                    elif sound["value"]["label"].startswith("Vacuum_cleaner"):
                        sound["value"]["label"] = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                sources.append(
                    {
                        "filename": source_file,
                        "onset": sound["value"]["event_time"],
                        "offset": sound["value"]["event_time"]
                        + sound["value"]["event_duration"],
                        "event_label": sound["value"]["label"],
                    }
                )

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"backgrounds": backgrounds, "sources": sources}, f, indent=4)

def gpu_decode_preds(strong_preds, thresholds, filenames, encoder, median_filter):
    # Init a dataframe per threshold
    bsz, cls, T = strong_preds.shape
    # Not implement for multiple thds
    thd = torch.cuda.FloatTensor(thresholds).reshape(-1, 1, 1, 1) # [Thds, Bsz, T, Cls]
    thd = thd.repeat(1, bsz, cls, T)
    binary_preds = strong_preds > thd
    smooth_preds = median_filter(binary_preds.float())
    prediction_dfs_gpu = encoder.gpu_decode_strong(smooth_preds, thresholds, filenames)
    return prediction_dfs_gpu