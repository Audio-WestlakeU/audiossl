import os
import threading
import multiprocessing
import sed_eval
import psds_eval

import numpy as np
import pandas as pd
from .psds import PSDSEval, plot_psd_roc

from concurrent.futures import ProcessPoolExecutor
from queue import Queue


g_manager = multiprocessing.Manager()


g_parallel = False

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
    label_interest=None,
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
    if label_interest is not None:
        gt_mask = [x in label_interest for x in gt["event_label"]]
        gt = gt[gt_mask]
        filenames = gt["filename"].values
        durations = durations[[x in filenames for x in durations["filename"]]]
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
            threshold_f1, f_dict = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_one_item(psds_eval: PSDSEval,prediction_dfs,i,k, weighted=False):
    det = prediction_dfs[k]
    # see issue https://github.com/audioanalytic/psds_eval/issues/3
    det["index"] = range(1, len(det) + 1)
    det = det.set_index("index")
    psds_args = psds_eval.add_operating_point_single_thread(
        det, info={"name": f"Op {i + 1:02d}", "threshold": k}, weighted=weighted
    )
    return psds_args

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
    label_interest=None,
    weighted=False
):
    print("Computing PSDS score ... (May take > 10 mins)")
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")

    if label_interest is not None:
        gt_mask = [x in label_interest for x in gt["event_label"]]
        gt = gt[gt_mask]
        filenames = gt["filename"].values
        durations = durations[[x in filenames for x in durations["filename"]]]
    
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    if g_parallel:
        # Parallel version (written by us)
        prediction_dfs=g_manager.dict(prediction_dfs)
        q = Queue(100)
        def helper_thread_fun():
            with ProcessPoolExecutor(max_workers=10) as exe:
                for i, k in enumerate(prediction_dfs.keys()):
                    q.put(exe.submit(compute_psds_one_item,psds_eval,prediction_dfs,i,k, weighted=weighted))
                q.put(None)

        helper_thread = threading.Thread(target=helper_thread_fun)
        helper_thread.setDaemon(True)
        helper_thread.start()

        for future in iter(q.get,""):
            if future is None:
                break
            else:
                psds_args=future.result()
                if psds_args is not None:
                    psds_eval._add_op(**psds_args)
    else:
        # Default behavior
        for i, k in enumerate(prediction_dfs.keys()):
            det = prediction_dfs[k]
            # see issue https://github.com/audioanalytic/psds_eval/issues/3
            det["index"] = range(1, len(det) + 1)
            det = det.set_index("index")
            psds_args = psds_eval.add_operating_point_single_thread(
                det, info={"name": f"Op {i + 1:02d}", "threshold": k}, weighted=weighted
            )
            if psds_args is not None:
                psds_eval._add_op(**psds_args)
    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr, weighted=weighted)


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
