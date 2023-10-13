import os
from audiossl.datasets.dcase_utils.encoder import ManyHotEncoder
import scipy

import numpy as np
import pandas as pd
import psds_eval
import sed_eval

from collections import OrderedDict
from pathlib import Path
from psds_eval import PSDSEval, plot_psd_roc

# Gloable variables following official codes
classes_labels = OrderedDict(
    {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }
)

def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
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

def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def psds_results(psds_obj):
    """ Compute psds scores
    Args:
        psds_obj: psds_eval.PSDSEval object with operating points.
    Returns:
    """
    try:
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
        print(f"\nPSD-Score (1, 0, 100): {psds_score.value:.5f}")
        psds_score = psds_obj.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
        print(f"\nPSD-Score (0, 1, 100): {psds_score.value:.5f}")
    except psds_eval.psds.PSDSEvalError as e:
        print("psds did not work ....")
        raise EnvironmentError


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric


def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment


def compute_per_intersection_macro_f1(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    """ Compute F1-score per intersection, using the defautl
    Args:
        prediction_dfs: dict, a dictionary with thresholds keys and predictions dataframe
        ground_truth_file: pd.DataFrame, the groundtruth dataframe
        durations_file: pd.DataFrame, the duration dataframe
        dtc_threshold: float, the parameter used in PSDSEval, percentage of tolerance for groundtruth intersection
            with predictions
        gtc_threshold: float, the parameter used in PSDSEval percentage of tolerance for predictions intersection
            with groundtruth
        gtc_threshold: float, the parameter used in PSDSEval to know the percentage needed to count FP as cross-trigger

    Returns:

    """
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")

    psds = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    psds_macro_f1 = []
    for threshold in prediction_dfs.keys():
        if not prediction_dfs[threshold].empty:
            threshold_f1, _ = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_from_operating_points(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

        plot_psd_roc(
            psds_score,
            filename=os.path.join(save_dir, f"PSDS_ct{alpha_ct}_st{alpha_st}_100.png"),
        )

    return psds_score.value


def pull_back_preds(preds, exp_len):
    exp_len = exp_len * 25
    cur_len = preds.shape[-1]
    # print("Expected length:", exp_len)
    # print("Prediction shape:", preds.shape)
    pull_preds = np.zeros((10, int(exp_len)))  # bsz, nclass, timestamps
    # print(preds.shape)
    # print(pull_preds.shape)
    for i, pred in enumerate(preds):
        temp = np.pad(pred, ((0, 0), (int(i * 200), int(exp_len - 200 * i - cur_len))))
        pull_preds += temp
    return pull_preds > 0

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
    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        for c_th in thresholds:
            c_preds = strong_preds[j]
            if pad_indx is not None:
                true_len = int(c_preds.shape[-1] * pad_indx[j].item())
                c_preds = c_preds[:true_len]
            pred = c_preds.transpose(0, 1).detach().cpu().numpy()
            pred = pred > c_th
            pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            prediction_dfs[c_th] = pd.concat([prediction_dfs[c_th], pred], ignore_index=True)

    return prediction_dfs
