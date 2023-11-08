# This code is the same as the PSDS in the psds_eval package
# We attempt to parallelize the code to speed up the evaluation

import typing
from warnings import warn
import pandas as pd
import numpy as np
import hashlib
from collections import namedtuple
import matplotlib.pyplot as plt


WORLD = "injected_psds_world_label"

RatesPerClass = namedtuple("RatesPerClass", ["tp_ratio", "fp_rate", "ct_rate",
                                             "effective_fp_rate", "id"])
PSDROC = namedtuple("PSDROC", ["yp", "xp", "mean", "std"])
PSDS = namedtuple("PSDS", ["value", "plt", "alpha_st", "alpha_ct", "max_efpr",
                           "duration_unit"])
Thresholds = namedtuple("Thresholds", ["gtc", "dtc", "cttc"])


from concurrent.futures import ProcessPoolExecutor
from queue import Queue
import threading
g_parallel=True

import time

def roc_curve_one_item(c,fpr_points,efpr_points,ctr_points,pcr,n_classes,_curve):
    tpr_v_fpr = _curve(fpr_points, pcr.fp_rate[c], pcr.tp_ratio[c])
    tpr_v_efpr = _curve(efpr_points, pcr.effective_fp_rate[c],
                        pcr.tp_ratio[c])
    tpr_v_ctr=np.full((n_classes,ctr_points.size),np.nan)
    for k in range(n_classes):
        if c == k:
            continue
        tpr_v_ctr[k]= _curve(ctr_points, pcr.ct_rate[c, k],
                                pcr.tp_ratio[c])

    return c,tpr_v_fpr,tpr_v_efpr,tpr_v_ctr


class PSDSEvalError(ValueError):
    """Error to be raised when function inputs are invalid"""
    pass


class PSDSEval:
    """A class to provide PSDS evaluation

    PSDS is the Polyphonic Sound Detection Score and was presented by
    Audio Analytic Labs in:
    A Framework for the Robust Evaluation of Sound Event Detection
    C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta, S. Krstulovic
    In IEEE International Conference on Acoustics, Speech, and Signal
    Processing (ICASSP). May 2020, URL: https://arxiv.org/abs/1910.08440

    Attributes:
        operating_points: An object containing all operating point data
        ground_truth: A pd.DataFrame that contains the ground truths
        metadata: A pd.DataFrame that contains the audio metadata
        class_names (list): A list of all class names in the evaluation
        threshold: (tuple): A namedTuple that contains the, gtc, dtc, and cttc
        nseconds (int): The number of seconds in the evaluation's unit of time
    """

    secs_in_uot = {"minute": 60, "hour": 3600, "day": 24 * 3600,
                   "month": 30 * 24 * 3600, "year": 365 * 24 * 3600}
    detection_cols = ["filename", "onset", "offset", "event_label"]

    def __init__(self, dtc_threshold=0.5, gtc_threshold=0.5,
                 cttc_threshold=0.3, **kwargs):
        """Initialise the PSDS evaluation

        Args:
            dtc_threshold: Detection Tolerance Criterion (DTC) threshold
            gtc_threshold: Ground Truth Intersection Criterion (GTC) threshold
            cttc_threshold: Cross-Trigger Tolerance Criterion (CTTC) threshold
            **kwargs:
            class_names: list of output class names. If not given it will be
                inferred from the ground truth table
            duration_unit: unit of time ('minute', 'hour', 'day', 'month',
                'year') for FP/CT rates report
            ground_truth (str): Path to the file containing ground truths.
            metadata (str): Path to the file containing audio metadata
        Raises:
            PSDSEvalError: If any of the input values are incorrect.
        """
        if dtc_threshold < 0.0 or dtc_threshold > 1.0:
            raise PSDSEvalError("dtc_threshold must be between 0 and 1")
        if cttc_threshold < 0.0 or cttc_threshold > 1.0:
            raise PSDSEvalError("cttc_threshold must be between 0 and 1")
        if gtc_threshold < 0.0 or gtc_threshold > 1.0:
            raise PSDSEvalError("gtc_threshold must be between 0 and 1")

        self.duration_unit = kwargs.get("duration_unit", "hour")
        if self.duration_unit not in self.secs_in_uot.keys():
            raise PSDSEvalError("Invalid duration_unit specified")
        self.nseconds = self.secs_in_uot[self.duration_unit]

        self.class_names = []
        self._update_class_names(kwargs.get("class_names", None))
        self.threshold = Thresholds(dtc=dtc_threshold, gtc=gtc_threshold,
                                    cttc=cttc_threshold)
        self.operating_points = self._operating_points_table()
        self.ground_truth = None
        self.metadata = None
        gt_t = kwargs.get("ground_truth", None)
        meta_t = kwargs.get("metadata", None)
        if gt_t is not None or meta_t is not None:
            self.set_ground_truth(gt_t, meta_t)
        n_cls_num = self._get_dataset_counts()
        class_names_set_no_world = set(self.class_names).difference([WORLD])
        self.cls_count_ratio = np.array(
            [n_cls_num[i] / n_cls_num.values.sum() for i, _ in enumerate(sorted(class_names_set_no_world))]
        )
        self.eval_call = 0
    @staticmethod
    def _validate_simple_dataframe(df, columns, name, allow_empty=False):
        """Validates given pandas.DataFrame

        Args:
            df (pandas.DataFrame): to be validated
            columns (list): Column names that should be in the df
            name (str): Name of the df. Only used when raising errors
            allow_empty (bool): If False then an empty df will raise an error
        Raises:
            PSDSEvalError: If the df provided is invalid
        """
        if not isinstance(df, pd.DataFrame):
            raise PSDSEvalError(f"The {name} data must be provided in "
                                "a pandas.DataFrame")
        if not set(columns).issubset(set(df.columns)):
            raise PSDSEvalError(f"The {name} data columns need to match the "
                                "following", columns)
        if not allow_empty and df.empty:
            raise PSDSEvalError(f"The {name} dataframe provided is empty")

    def _validate_input_table_with_events(self, df, name):
        """Validates given pandas.DataFrame with events

        Args:
            df (pandas.DataFrame): to be validated
            name (str): Name of the df. Only used when raising errors
        Raises:
            PSDSEvalError: If the df provided is invalid, has overlapping
                events from the same class or has offset happening after onset.
        """
        self._validate_simple_dataframe(
            df, self.detection_cols, name, allow_empty=True
        )
        if not df.empty:
            if not df[df.onset > df.offset].empty:
                raise PSDSEvalError(f"The {name} dataframe provided has "
                                    f"events with onset > offset.")
            intersections = self._get_table_intersections(
                df, df, suffixes=("_1", "_2"), remove_identical=True)
            if not intersections[intersections.same_cls].empty:
                raise PSDSEvalError(f"The {name} dataframe provided has "
                                    f"intersecting events/labels for the "
                                    f"same class.")

    def num_operating_points(self):
        """Returns the number of operating point registered"""
        return len(self.operating_points.id)

    def _update_class_names(self, new_classes: list):
        """Adds new class names to the existing set

        Updates unique class names and merges them with existing class_names
        """
        if new_classes is not None and len(new_classes) > 0:
            new_classes = set(new_classes)
            _classes = set(self.class_names)
            _classes.update(new_classes)
            self.class_names = sorted(_classes)

    def set_ground_truth(self, gt_t, meta_t):
        """Validates and updates the class with a set of Ground Truths

        The Ground Truths and Metadata are used to count true positives
        (TPs), false positives (FPs) and cross-triggers (CTs) for all
        operating points when they are later added.

        Args:
            gt_t (pandas.DataFrame): A table of ground truths
            meta_t (pandas.DataFrame): A table of audio metadata information

        Raises:
            PSDSEvalError if there is an issue with the input data

        """
        if self.ground_truth is not None or self.metadata is not None:
            raise PSDSEvalError("You cannot set the ground truth more than"
                                " once per evaluation")
        if gt_t is None and meta_t is not None:
            raise PSDSEvalError("The ground truth cannot be set without data")
        if meta_t is None and gt_t is not None:
            raise PSDSEvalError("Audio metadata is required when adding "
                                "ground truths")

        self._validate_input_table_with_events(gt_t, "ground truth")
        self._validate_simple_dataframe(
            meta_t, ["filename", "duration"], "metadata")

        # re-indexing is done to protect against duplicate indexes
        _ground_truth = gt_t[self.detection_cols].reset_index(
            inplace=False, drop=True
        )
        _metadata = meta_t.reset_index(inplace=False, drop=True)

        # remove duplicated entries (possible mistake in its generation?)
        _metadata = _metadata.drop_duplicates("filename")
        metadata_t = _metadata.sort_values(by=["filename"], axis=0)
        _ground_truth = self._update_world_detections(self.detection_cols,
                                                      _ground_truth,
                                                      metadata_t)
        # remove zero-length events
        _ground_truth = _ground_truth[_ground_truth.offset >
                                      _ground_truth.onset]
        ground_truth_t = _ground_truth.sort_values(by=self.detection_cols[:2],
                                                   axis=0)
        ground_truth_t.dropna(inplace=True)
        ground_truth_t["duration"] = \
            ground_truth_t.offset - ground_truth_t.onset
        ground_truth_t["id"] = ground_truth_t.index

        self._update_class_names(ground_truth_t.event_label)
        self.ground_truth = ground_truth_t
        self.metadata = metadata_t

    def _init_det_table(self, det_t):
        """Validate and prepare an input detection table

        Validates and updates the a detection table with an 'id' and
        duration column.

        Args:
            det_t (pandas.DataFrame): A system's detection table

        Returns:
            A tuple with the three validated and processed tables
        """
        self._validate_input_table_with_events(
            det_t, "detection")
        # we re-index the detection table in case invalid indexes (indexes
        # with repeated index values) are given
        det_t = det_t.reset_index(inplace=False, drop=True)
        # remove zero-length or invalid events
        det_t = det_t[det_t.offset > det_t.onset]
        detection_t = det_t.sort_values(by=self.detection_cols[:2], axis=0)
        detection_t["duration"] = detection_t.offset - detection_t.onset
        detection_t["id"] = detection_t.index
        return detection_t

    @staticmethod
    def _update_world_detections(columns, ground_truth, metadata):
        """Extend the ground truth with WORLD detections

        Append to each file an artificial ground truth of length equal
        to the file duration provided in the metadata table.
        """
        world_gt = [
            {k: v for k, v in zip(columns,
                                  [metadata.loc[i, 'filename'], 0.0,
                                   metadata.loc[i, 'duration'], WORLD])
             } for i in metadata.index
        ]
        if len(world_gt):
            ground_truth = pd.concat(
                [ground_truth, pd.DataFrame(world_gt)], ignore_index=True)
        return ground_truth

    def _operating_point_id(self, detection_table):
        """Used to produce a unique ID for each operating point

        here we sort the dataframe so that shuffled versions of the same
        detection table results in the same hash
        """

        table_columns = ["filename", "onset", "offset", "event_label"]
        detection_table_col_sorted = detection_table[
            table_columns]
        detection_table_row_sorted = detection_table_col_sorted.sort_values(
            by=table_columns)
        h = hashlib.sha256(pd.util.hash_pandas_object(
            detection_table_row_sorted, index=False).values)
        uid = h.hexdigest()
        if uid in self.operating_points.id.values:
            warn("A similar operating point exists, skipping this one")
            uid = ""
        return uid

    @staticmethod
    def _get_table_intersections(table1, table2, suffixes=("_1", "_2"),
                                 remove_identical=False):
        """Creates a table of intersecting events/labels in two tables

        Returns:
            A pandas table with intersecting events with columns of given
            suffixes from each input table. A boolean "same_cls" column
            indicates if intersecting events have the same class. If
            remove_identical=True, identical events from both tables are not
            considered.
        """
        comb_t = pd.merge(table1, table2,
                          how='outer', on='filename',
                          suffixes=suffixes)
        # intersect_t contains detections/labels in the first table that
        # intersect one or more detections/labels in the second table
        # with non-zero intersections
        intersect_t = comb_t[
            (comb_t["onset" + suffixes[0]] < comb_t["offset" + suffixes[1]]) &
            (comb_t["onset" + suffixes[1]] < comb_t["offset" + suffixes[0]]) &
            comb_t.filename.notna()].copy(deep=True)
        if remove_identical:
            intersect_t = intersect_t[
                (intersect_t["onset" + suffixes[0]] !=
                 intersect_t["onset" + suffixes[1]]) |
                (intersect_t["offset" + suffixes[1]] !=
                 intersect_t["offset" + suffixes[0]]) |
                (intersect_t["event_label" + suffixes[0]] !=
                 intersect_t["event_label" + suffixes[1]])]
        # Add a flag to show that labels/events from the first and second
        # tables are of the same class
        intersect_t["same_cls"] = (
                intersect_t["event_label" + suffixes[0]] ==
                intersect_t["event_label" + suffixes[1]])
        return intersect_t

    def _ground_truth_intersections(self, detection_t, ground_truth_t):
        """Creates a table to represent the ground truth intersections

        Returns:
            A pandas table that contains the following columns:
                inter_duration: intersection between detection and gt (s)
                det_precision: indicates what portion of a detection
                    intersect one or more ground truths of the same class
                gt_coverage: measures what proportion of a ground truth
                    is covered by one or more detections of the same class
        """
        cross_t = self._get_table_intersections(
            detection_t, ground_truth_t, suffixes=("_det", "_gt"))

        cross_t["inter_duration"] = \
            np.minimum(cross_t.offset_det, cross_t.offset_gt) - \
            np.maximum(cross_t.onset_det, cross_t.onset_gt)
        cross_t["det_precision"] = \
            cross_t.inter_duration / cross_t.duration_det
        cross_t["gt_coverage"] = \
            cross_t.inter_duration / cross_t.duration_gt
        return cross_t

    def _detection_and_ground_truth_criteria(self, cross_t):
        """Creates GTC and DTC detection sets

        Args:
            cross_t (pandas.DataFrame): A DataFrame containing detections and
                their timings that intersect with the class's ground truths.

        Returns:
            A tuple that contains two DataFrames. The first a table of
            true positive detections that satisfy both DTC and GTC. The
            second contains only the IDs of the detections that satisfy
            the DTC.
        """

        # Detections that intersect with the the ground truths
        gt_cross_t = cross_t[cross_t.same_cls]

        # Group the duplicate detections and sum the det_precision
        if gt_cross_t.empty:
            dtc_t = pd.DataFrame(columns=["id_det", "event_label_gt",
                                          "det_precision"])
        else:
            dtc_t = gt_cross_t.groupby(
                ["id_det", "event_label_gt"]
            ).det_precision.sum().reset_index()

        dtc_ids = dtc_t[dtc_t.det_precision >= self.threshold.dtc].id_det

        # Group the duplicate detections that exist in the DTC set and sum
        gtc_t = gt_cross_t[gt_cross_t.id_det.isin(dtc_ids)].groupby(
            ["id_gt", "event_label_det"]
        ).gt_coverage.sum().reset_index()

        # Join the two into a single true positive table
        if len(dtc_t) or len(gtc_t):
            tmp = pd.merge(gt_cross_t, dtc_t, on=["id_det", "event_label_gt"],
                           suffixes=("", "_sum")
                           ).merge(gtc_t, on=["id_gt", "event_label_det"],
                                   suffixes=("", "_sum"))
        else:
            cols = gt_cross_t.columns.to_list() + \
                   ["det_precision_sum", "gt_coverage_sum"]
            tmp = pd.DataFrame(columns=cols)

        dtc_filter = tmp.det_precision_sum >= self.threshold.dtc
        gtc_filter = tmp.gt_coverage_sum >= self.threshold.gtc
        return tmp[dtc_filter & gtc_filter], dtc_ids

    def _get_dataset_duration(self):
        """Compute duraion of on the source data.

        Compute the duration per class, and total duration for false
        positives."""
        t_filter = self.ground_truth.event_label == WORLD
        data_duration = self.ground_truth[t_filter].duration.sum()
        gt_durations = self.ground_truth.groupby("event_label").duration.sum()
        return gt_durations, data_duration

    def _get_dataset_counts(self):
        """Compute event counts on the source data.

        Compute the number of events per class."""
        gt_counts = self.ground_truth.groupby("event_label").filename.count()
        return gt_counts

    def _confusion_matrix_and_rates(self, tp, ct):
        """Produces the confusion matrix and per-class detection rates.

        The first dimension of the confusion matrix (axis 0) represents the
        system detections, while the second dimension (axis 1) represents the
        ground truths.

        Args:
            tp (pandas.DataFrame): table of true positive detections that
                satisfy both the DTC and GTC
            ct (pandas.DataFrame): table with cross-triggers (detections that
                satisfy the CTTC)

        Returns:
            A tuple with confusion matrix, true positive ratios, false
            positive rates and cross-trigger rates. Note that the
            cross-trigger rate array will contain NaN values along its
            diagonal.

        """
        n_real_classes = len(self.class_names) - 1  # don't count WORLD
        counts = np.zeros([len(self.class_names), len(self.class_names)])
        tp_ratio = np.empty(n_real_classes)
        fp_rate = np.empty(n_real_classes)
        ct_rate = np.full((n_real_classes, n_real_classes), np.nan)
        # Create an ordered set of class names without world
        class_names_set_no_world = set(self.class_names).difference([WORLD])
        # Create an ordered set of class names with world at the end
        cls_names_world_end = sorted(class_names_set_no_world)
        cls_names_world_end.append(WORLD)
        # Multi-indexed pandas.Series. Values are simply the overall item count
        ct_tmp = ct.groupby(["event_label_det",
                             "event_label_gt"]).filename.count()
        n_cls_gt = self._get_dataset_counts()
        gt_dur, dataset_dur = self._get_dataset_duration()
        # i, cls: detection -- j, ocls: ground truth
        for i, cls in enumerate(sorted(class_names_set_no_world)):
            counts[i, i] = len(tp[tp.event_label_gt == cls])
            if cls in n_cls_gt:
                tp_ratio[i] = counts[i, i] / n_cls_gt[cls]
            for j, ocls in enumerate(cls_names_world_end):
                try:
                    counts[i, j] = ct_tmp[cls, ocls]
                except KeyError:
                    pass
                if ocls == WORLD:
                    fp_rate[i] = counts[i, j] * self.nseconds / dataset_dur
                elif j != i:
                    ct_rate[i, j] = counts[i, j] * self.nseconds / gt_dur[ocls]
        return counts, tp_ratio, fp_rate, ct_rate

    def _cross_trigger_criterion(self, inter_t, tp_t, dtc_ids):
        """Produce a set of detections that satisfy the CTTC

        Using the main intersection table and output from the dtc function. A
        set of False Positive Cross-Triggered detections is made and then
        filtered by the CTTC threshold.

        The CTTC set consists of detections that:
            1) are not in the True Positive table
            2) intersect with ground truth of a different class (incl. WORLD)
            3) have not satisfied the detection tolerance criterion

        Args:
            inter_t (pandas.DataFrame): The table of detections and their
                ground truth intersection calculations
            tp_t (pandas.DataFrame): A detection table containing true positive
                detections.
            dtc_ids (pandas.DataFrame): A table containing a list of the uid's
                that pass the dtc.
        """

        ct_t = inter_t[~inter_t.id_det.isin(tp_t.id_det) &
                       ~inter_t.same_cls & ~inter_t.id_det.isin(dtc_ids)]

        # Group the duplicate detections and sum
        tmp = ct_t.groupby(["id_det", "event_label_gt"]).det_precision.sum()
        if len(tmp):
            ct_t = pd.merge(ct_t, tmp.reset_index(), suffixes=("", "_sum"),
                            on=["id_det", "event_label_gt"])
        else:
            ct_t["det_precision_sum"] = 0.0

        # Ensure that all world events are also collected
        cttc = ct_t[(ct_t.det_precision_sum >= self.threshold.cttc) |
                    (ct_t.event_label_gt == WORLD)]
        return cttc

    def _evaluate_detections(self, det_t):
        """
        Apply the DTC/GTC/CTTC definitions presented in the ICASSP paper (link
        above) to computes the confusion matrix and the per-class true positive
        ratios, false positive rates and cross-triggers rates.

        Args:
            det_t: (pandas.DataFrame): An initialised detections table

        Returns:
            tuple containing confusion matrix, TP_ratio, FP_rate and CT_rate
        """

        inter_t = self._ground_truth_intersections(det_t, self.ground_truth)
        tp, dtc_ids = self._detection_and_ground_truth_criteria(inter_t)
        cttc = self._cross_trigger_criterion(inter_t, tp, dtc_ids)

        # For the final detection count we must drop duplicates
        cttc = cttc.drop_duplicates(["id_det", "event_label_gt"])
        tp = tp.drop_duplicates("id_gt")

        cts, tp_ratio, fp_rate, ct_rate = \
            self._confusion_matrix_and_rates(tp, cttc)

        return cts, tp_ratio, fp_rate, ct_rate


    def add_operating_point_single_thread(self, detections, info=None):
        if self.ground_truth is None:
            raise PSDSEvalError("Ground Truth must be provided before "
                                "adding the first operating point")
        if self.metadata is None:
            raise PSDSEvalError("Audio metadata must be provided before "
                                "adding the first operating point")

        # validate and prepare tables
        det_t = self._init_det_table(detections)
        op_id = self._operating_point_id(det_t)
        if not op_id:
            return

        cts, tp_ratio, fp_rate, ct_rate = self._evaluate_detections(det_t)
        return {"opid": op_id, 
                "counts": cts, 
                "tpr": tp_ratio, 
                "fpr": fp_rate,
                "ctr": ct_rate, 
                "info": info}

    def add_operating_point(self, detections, info=None):
        """Adds a new Operating Point (OP) into the evaluation

        An operating point is defined by a system's detection results given
        some user parameters. It is expected that a user generates detection
        data from multiple operating points and then passes all data to this
        function during a single system evaluation so that a comprehensive
        result can be provided.

        Args:
            detections (pandas.DataFrame): A table of system detections
                that has the following columns:
                "filename", "onset", "offset", "event_label".
            info (dict): A dictionary of optional information associated
                with the operating point, used for keeping track of
                how the operating point is generated by the user
        Raises:
            PSDSEvalError: If the PSDSEval ground_truth or metadata are unset.
        """
        if self.ground_truth is None:
            raise PSDSEvalError("Ground Truth must be provided before "
                                "adding the first operating point")
        if self.metadata is None:
            raise PSDSEvalError("Audio metadata must be provided before "
                                "adding the first operating point")

        # validate and prepare tables
        det_t = self._init_det_table(detections)
        op_id = self._operating_point_id(det_t)
        if not op_id:
            return

        cts, tp_ratio, fp_rate, ct_rate = self._evaluate_detections(det_t)
        self._add_op(opid=op_id, counts=cts, tpr=tp_ratio, fpr=fp_rate,
                     ctr=ct_rate, info=info)

    @staticmethod
    def _operating_points_table():
        """Returns and empty operating point table with the correct columns"""
        return pd.DataFrame(columns=["id", "counts", "tpr", "fpr", "ctr"])

    def _add_op(self, opid, counts, tpr, fpr, ctr, info=None):
        """Adds a new operating point into the class"""
        op = {"id": opid, "counts": counts, "tpr": tpr, "fpr": fpr, "ctr": ctr}
        if not info:
            info = dict()

        if set(op.keys()).isdisjoint(set(info.keys())):
            op.update(info)
            self.operating_points = pd.concat(
                [self.operating_points, pd.DataFrame([op])], ignore_index=True)
        else:
            raise PSDSEvalError("the 'info' cannot contain the keys 'id', "
                                "'counts', 'tpr', 'fpr' or 'ctr'")

    def clear_all_operating_points(self):
        """Deletes any Operating Point previously added. An evaluation of new
        OPs can be safely performed once this function is executed.

        Note that neither the task definition (i.e. self.threshold) nor the
        dataset (i.e. self.metadata and self.ground_truth) are affected by this
        function.
        """
        del self.operating_points
        self.operating_points = self._operating_points_table()

    @staticmethod
    def perform_interp(x, xp, yp):
        """Interpolate the curve (xp, yp) over the points given in x

        This interpolation function uses numpy.interp but deals with
        duplicates in xp quietly.

        Args:
            x (numpy.ndarray): a series of points at which to
                evaluate the interpolated values
            xp (numpy.ndarray): x-values of the curve to be interpolated
            yp (numpy.ndarray): y-values of the curve to be interpolated

        Returns:
            Interpolated values stored in a numpy.ndarray
        """
        new_y = np.zeros_like(x)
        sorted_idx = np.argsort(xp)
        xp_unq, idx = np.unique(xp[sorted_idx], return_index=True)
        valid_x = x < xp_unq[-1]
        new_y[valid_x] = np.interp(x[valid_x], xp_unq, yp[sorted_idx][idx])
        # fill remaining point with last tp value
        last_value = yp[sorted_idx][idx[-1]]
        new_y[~valid_x] = last_value
        # make monotonic
        new_y = np.maximum.accumulate(new_y)
        return new_y

    @staticmethod
    def step_curve(x, xp, yp):
        """Performs a custom interpolation on the ROC described by (xp, yp)

        The interpolation is performed on the given x-coordinates (x)
        and x.size >= unique(xp).size. If more than one yp value exists
        for the same xp value, only the highest yp is retained. Also yp
        is made non-decreasing so that sub optimal operating points are
        ignored.

        Args:
            x (numpy.ndarray): a series of points at which to
                evaluate the interpolated values
            xp (numpy.ndarray): x-values of the curve to be interpolated
            yp (numpy.ndarray): y-values of the curve to be interpolated

        Returns:
            numpy.ndarray: An array of interpolated y values
        """
        roc_orig = pd.DataFrame({'x': xp, 'y': yp})
        roc_valid_only = (roc_orig.groupby('x')
                          .agg('max')
                          .reset_index()
                          .sort_values(by='x'))
        if x.size < roc_valid_only.x.size:
            raise PSDSEvalError(f"x: {x.size}, xp: {xp.size}")
        # make y monotonic (given the TP/FP counting method rocs are not
        # monotonically increasing)
        roc_valid_only.y = roc_valid_only.y.cummax()
        roc_new = pd.merge(
            pd.Series(x, name='x'),
            roc_valid_only,
            how="outer",
            on="x").fillna(method='ffill').fillna(value=0)
        return roc_new.y.values

    def _effective_fp_rate(self, alpha_ct=0.):
        """Calculates effective False Positive rate (eFPR)

        Calculates the the eFPR per class applying the given weight
        to cross-triggers.

        Args:
             alpha_ct (float): cross-trigger weight in effective
                 FP rate computation
        """
        if alpha_ct < 0 or alpha_ct > 1:
            raise PSDSEvalError("alpha_ct must be between 0 and 1")

        # add a zero-point in each arr below (using np.pad)
        tpr_arr = np.stack(self.operating_points.tpr.values, axis=1)
        tpr_arr = np.pad(tpr_arr, ((0, 0), (0, 1)), "constant",
                         constant_values=0)
        fpr_arr = np.stack(self.operating_points.fpr.values, axis=1)
        fpr_arr = np.pad(fpr_arr, ((0, 0), (0, 1)), "constant",
                         constant_values=0)
        ctr_arr = np.stack(self.operating_points.ctr.values, axis=2)
        ctr_arr = np.pad(ctr_arr, ((0, 0), (0, 0), (0, 1)), "constant",
                         constant_values=0)
        id_arr = self.operating_points.id.values
        id_arr = np.pad(id_arr, ((0, 1),), "constant", constant_values="None")

        efpr = fpr_arr + alpha_ct * np.nanmean(ctr_arr, axis=1)

        return RatesPerClass(tp_ratio=tpr_arr, fp_rate=fpr_arr,
                             ct_rate=ctr_arr, effective_fp_rate=efpr,
                             id=id_arr)

    def psd_roc_curves(self, alpha_ct, linear_interp=False):
        """Generates PSD-ROC TPR vs FPR/eFPR/CTR

        Args:
            alpha_ct (float): The weighting placed upon cross triggered FPs
            linear_interp (bool): Enables linear interpolation.

        Returns:
            A tuple containing the following ROC curves, tpr_vs_fpr,
            tpr_vs_ctr, tpr_vs_efpr.
        """
        pcr = self._effective_fp_rate(alpha_ct)
        n_classes = len(self.class_names) - 1
        # common x-axis built as union of points across classes
        fpr_points = np.unique(np.sort(pcr.fp_rate.flatten()))
        efpr_points = np.unique(np.sort(pcr.effective_fp_rate.flatten()))
        ctr_points = np.unique(np.sort(pcr.ct_rate.flatten()))
        tpr_v_fpr = np.full((n_classes, fpr_points.size), np.nan)
        tpr_v_efpr = np.full((n_classes, efpr_points.size), np.nan)
        tpr_v_ctr = np.full((n_classes, n_classes, ctr_points.size), np.nan)
        _curve = self.perform_interp if linear_interp else self.step_curve
        if g_parallel:
            pass
            q = Queue(100)
            def helper_thread_fun():
                with ProcessPoolExecutor(max_workers=10) as exe:

                    for c in range(n_classes):
                        q.put(exe.submit(roc_curve_one_item,c,fpr_points,efpr_points,ctr_points,pcr,n_classes,_curve))
                    q.put(None)

            helper_thread = threading.Thread(target=helper_thread_fun)
            helper_thread.setDaemon(True)
            helper_thread.start()


            for future in iter(q.get,""):
                if future is None:
                    break
                else:
                    c,tpr_v_fpr_c,tpr_v_efpr_c,tpr_v_ctr_c=future.result()
                    tpr_v_fpr[c] = tpr_v_fpr_c
                    tpr_v_efpr[c] = tpr_v_efpr_c
                    for k in range(n_classes):
                        tpr_v_ctr[c,k] = tpr_v_ctr_c[k]
        else:
            for c in range(n_classes):
                tpr_v_fpr[c] = _curve(fpr_points, pcr.fp_rate[c], pcr.tp_ratio[c])
                tpr_v_efpr[c] = _curve(efpr_points, pcr.effective_fp_rate[c],
                                    pcr.tp_ratio[c])
                for k in range(n_classes):
                    if c == k:
                        continue
                    tpr_v_ctr[c, k] = _curve(ctr_points, pcr.ct_rate[c, k],
                                            pcr.tp_ratio[c])

        tpr_vs_fpr_c = PSDROC(
            yp=tpr_v_fpr, xp=fpr_points,
            mean=np.nanmean(tpr_v_fpr, axis=0),
            std=np.nanstd(tpr_v_fpr, axis=0)
        )
        tpr_vs_efpr_c = PSDROC(
            yp=tpr_v_efpr, xp=efpr_points,
            mean=np.nanmean(tpr_v_efpr, axis=0),
            std=np.nanstd(tpr_v_efpr, axis=0)
        )
        tpr_vs_ctr_c = PSDROC(
            yp=tpr_v_ctr, xp=ctr_points,
            mean=np.nanmean(tpr_v_ctr.reshape([-1, ctr_points.size]), axis=0),
            std=np.nanstd(tpr_v_ctr.reshape([-1, ctr_points.size]), axis=0)
        )
        return tpr_vs_fpr_c, tpr_vs_ctr_c, tpr_vs_efpr_c

    @staticmethod
    def compute_f_score(TP_values, FP_values, FN_values, beta):
        """Computes the F-scores for the given TP/FP/FN"""
        k = (1 + beta ** 2)
        f_scores = (k * TP_values) / (k * TP_values + (beta ** 2) * FN_values
                                      + FP_values)
        return f_scores

    def compute_macro_f_score(self, detections, beta=1.):
        """Computes the macro F_score for the given detection table

        The DTC/GTC/CTTC criteria presented in the ICASSP paper (link above)
        are exploited to compute the confusion matrix. From the latter, class
        dependent F_score metrics are computed. These are further averaged to
        compute the macro F_score.

        It is important to notice that a cross-trigger is also counted as
        false positive.

        Args:
            detections (pandas.DataFrame): A table of system detections
                that has the following columns:
                "filename", "onset", "offset", "event_label".
            beta: coefficient used to put more (beta > 1) or less (beta < 1)
                emphasis on false negatives.

        Returns:
            A tuple with average F_score and dictionary with per-class F_score

        Raises:
            PSDSEvalError: if class instance doesn't have ground truth table
        """
        if self.ground_truth is None:
            raise PSDSEvalError("Ground Truth must be provided before "
                                "adding the first operating point")

        det_t = self._init_det_table(detections)
        counts, tp_ratios, _, _ = self._evaluate_detections(det_t)

        per_class_tp = np.diag(counts)[:-1]
        num_gts = per_class_tp / tp_ratios
        per_class_fp = counts[:-1, -1]
        per_class_fn = num_gts - per_class_tp
        f_per_class = self.compute_f_score(per_class_tp, per_class_fp,
                                           per_class_fn, beta)

        # remove the injected world label
        class_names_no_world = sorted(set(self.class_names
                                          ).difference([WORLD]))
        f_dict = {c: f for c, f in zip(class_names_no_world, f_per_class)}
        f_avg = np.nanmean(f_per_class)

        return f_avg, f_dict

    def select_operating_points_per_class(self, class_constraints,
                                          alpha_ct=0., beta=1.):
        """Returns the operating points for given constraints.

        Finds the operating points which best satisfy the requested
        constraints per class. For the "tpr" constraint, the operating point
        with the lowest eFPR among the ones with TPR greater than or equal
        to the given value will be returned.
        Similarly, for the "fpr" and "efpr" constraints, the operating point
        with the highest possible TPR among the ones with FPR or eFPR lower
        than or equal to the given constraint will be returned.
        For the "fscore" constraint, the operating point with the highest
        fscore is returned.

        If the desired operating constraint is not achievable, the
        corresponding row in the returned table has np.nan values.

        Args:
            class_constraints (pandas.DataFrame): A table of operating point
                requirement descriptions per class, with the columns
                "class_name", "constraint", "value":
                "class_name": is the name of the class for the constraint
                "constraint": is be one of "tpr", "fpr", "efpr", "fscore",
                "value": is the desired value for the given constraint type.
                         If the constraint is "fscore", the value field is
                         ignored.
            alpha_ct (float): cross-trigger weight in effective FP
                rate computation
            beta: the parameter for the F-score (F1-score is computed by
                default)

        Returns:
            A table of operating point information for each given consraint.
            If no operating point satisfies the requested constraint,
            a row of NaN values is returned instead.

        Raises:
            PSDSEvalError: If there is an issue with the class_constraints
                table
        """

        self._validate_simple_dataframe(
            class_constraints,
            columns=["class_name", "constraint", "value"],
            name="constraints")
        pcr = self._effective_fp_rate(alpha_ct)
        class_names_no_world = sorted(
            set(self.class_names).difference([WORLD]))
        _op_points_t_cols = ["class_name", "TPR", "FPR", "eFPR", "Fscore"]
        _op_points_t_cols.extend(self.operating_points.columns)
        for col in ["id", "counts", "tpr", "fpr", "ctr"]:
            _op_points_t_cols.remove(col)
        _op_points: typing.List[pd.DataFrame] = list()

        n_cls_gt = self._get_dataset_counts()
        gt_dur, dataset_dur = self._get_dataset_duration()

        for index, row in class_constraints.iterrows():
            class_name = row["class_name"]
            if class_name not in self.class_names:
                raise PSDSEvalError(f"Unknown class: {class_name}")
            class_index = class_names_no_world.index(class_name)
            value = row["value"]
            constraint = row["constraint"]
            efpr = pcr.effective_fp_rate[class_index]
            tpr = pcr.tp_ratio[class_index]
            fpr = pcr.fp_rate[class_index]

            tp_counts = tpr * n_cls_gt[class_name]
            fn_counts = n_cls_gt[class_name] - tp_counts
            fp_counts = fpr * dataset_dur / self.nseconds
            f_scores = self.compute_f_score(tp_counts, fp_counts,
                                            fn_counts, beta=beta)

            op_index = None
            if constraint == "tpr":
                _filter = tpr >= value
                _filter_arr = efpr
                _filter_op = np.argmin
            elif constraint == "fpr":
                _filter = fpr <= value
                _filter_arr = tpr
                _filter_op = np.argmax
            elif constraint == "efpr":
                _filter = efpr <= value
                _filter_arr = tpr
                _filter_op = np.argmax
            elif constraint == "fscore":
                _filter = np.ones(tpr.shape, dtype=bool)
                _filter_arr = f_scores
                _filter_op = np.argmax
            else:
                raise PSDSEvalError(f"The constraint has to be one of tpr, "
                                    f"fpr, efpr or fscore, instead it is"
                                    f" {constraint}")
            if _filter.sum() > 0:
                op_index = _filter_op(_filter_arr[_filter]).flatten()[0]
            if (op_index is None) or (pcr.id[_filter][op_index] == "None"):
                chosen_op_point_dict = {"class_name": class_name,
                                        "TPR": np.nan,
                                        "FPR": np.nan,
                                        "eFPR": np.nan,
                                        "Fscore": np.nan}
            else:
                id_selected = pcr.id[_filter][op_index]
                chosen_op_point_dict = {"class_name": class_name,
                                        "TPR": tpr[_filter][op_index],
                                        "FPR": fpr[_filter][op_index],
                                        "eFPR": efpr[_filter][op_index],
                                        "Fscore": f_scores[_filter][op_index]}
                chosen_op_point = self.operating_points[
                    self.operating_points["id"] == id_selected][
                    _op_points_t_cols[5:]]
                chosen_op_point_dict.update(
                    chosen_op_point.to_dict(orient="records")[0])
            _op_points.append(pd.DataFrame(
                [chosen_op_point_dict], columns=_op_points_t_cols))

        return pd.concat(_op_points, ignore_index=True)

    def _effective_tp_ratio(self, tpr_efpr, alpha_st):
        """Calculates the effective true positive rate (eTPR)

        Reduces a set of class ROC curves into a single Polyphonic
        Sound Detection (PSD) ROC curve. If NaN values are present they
        will be converted to zero.

        Args:
            tpr_efpr (PSDROC): A ROC that describes the PSD-ROC for
                all classes
            alpha_st (float): A weighting applied to the
                inter-class variability

        Returns:
            PSDROC: A namedTuple that describes the PSD-ROC used for the
                calculation of PSDS.
        """
        etpr = tpr_efpr.mean - alpha_st * tpr_efpr.std
        np.nan_to_num(etpr, copy=False, nan=0.0)
        etpr = np.where(etpr < 0, 0.0, etpr)
        return PSDROC(xp=tpr_efpr.xp, yp=etpr, std=tpr_efpr.std,
                    mean=tpr_efpr.mean)

    def psds(self, alpha_ct=0.0, alpha_st=0.0, max_efpr=None, en_interp=False):
        """Computes PSDS metric for given system

        Args:
            alpha_ct (float): cross-trigger weight in effective FP
                rate computation
            alpha_st (float): cost of instability across classes used
                to compute effective TP ratio (eTPR). Must be positive
            max_efpr (float): maximum effective FP rate at which the SED
                system is evaluated (default: 100 errors per unit of time)
            en_interp (bool): if true the psds is calculated using
                linear interpolation instead of a standard staircase
                when computing PSD ROC

        Returns:
            A (PSDS) Polyphonic Sound Event Detection Score object
        """
        if alpha_st < 0:
            raise PSDSEvalError("alpha_st can't be negative")

        tpr_fpr_curve, tpr_ctr_curve, tpr_efpr_curve = \
            self.psd_roc_curves(alpha_ct, en_interp)

        if max_efpr is None:
            max_efpr = np.max(tpr_efpr_curve.xp)

        psd_roc = self._effective_tp_ratio(tpr_efpr_curve, alpha_st)
        score = self._auc(psd_roc.xp, psd_roc.yp, max_efpr,
                          alpha_st > 0) / max_efpr
        return PSDS(value=score, plt=psd_roc, alpha_st=alpha_st,
                    alpha_ct=alpha_ct, max_efpr=max_efpr,
                    duration_unit=self.duration_unit)

    @staticmethod
    def _auc(x, y, max_x=None, decreasing_y=False):
        """Compute area under curve described by the given x, y points.

        To avoid an overestimate the area in case of large gaps between
        points, the area is computed as sums of rectangles rather than
        trapezoids (np.trapz).

        Both x and y must be non-decreasing 1-dimensional numpy.ndarray. In
        particular cases it is necessary to relax such constraint for y. This
        can be done by setting allow_decrease_y to True.
        The non-decreasing property is verified if
        for all i in {2, ..., x.size}, x[i-1] <= x[i]

        Args:
            x (numpy.ndarray): 1-D array containing non-decreasing
                values for x-axis
            y (numpy.ndarray): 1-D array containing non-decreasing
                values for y-axis
            max_x (float): maximum x-coordinate for area computation
            decreasing_y (bool): controls the check for non-decreasing property
                of y

        Returns:
             A float that represents the area under curve

        Raises:
            PSDSEvalError: If there is an issue with the input data
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise PSDSEvalError("x and y must be provided as a numpy.ndarray")
        if x.ndim > 1 or y.ndim > 1:
            raise PSDSEvalError("x or y are not 1-dimensional numpy.ndarray")
        if x.size != y.size:
            raise PSDSEvalError(f"x and y must be of equal "
                                f"length {x.size} != {y.size}")
        if np.any(np.diff(x) < 0):
            raise PSDSEvalError("non-decreasing property not verified for x")
        if not decreasing_y and np.any(np.diff(y) < 0):
            raise PSDSEvalError("non-decreasing property not verified for y")
        _x = np.array(x)
        _y = np.array(y)

        if max_x is None:
            max_x = _x.max()
        if max_x not in _x:
            # add max_x to x and the correspondent y value
            _x = np.sort(np.concatenate([_x, [max_x]]))
            max_i = int(np.argwhere(_x == max_x))
            _y = np.concatenate([_y[:max_i], [_y[max_i-1]], _y[max_i:]])
        valid_idx = _x <= max_x
        dx = np.diff(_x[valid_idx])
        _y = np.array(_y[valid_idx])[:-1]
        if dx.size != _y.size:
            raise PSDSEvalError(f"{dx.size} != {_y.size}")
        return np.sum(dx * _y)


def plot_psd_roc(psd, en_std=False, axes=None, filename=None, **kwargs):
    """Shows (or saves) the PSD-ROC with optional standard deviation.

    When the plot is generated the area under PSD-ROC is highlighted.
    The plot is affected by the values used to compute the metric:
    max_efpr, alpha_ST and alpha_CT

    Args:
        psd (PSDS): The psd_roc that is to be plotted
        en_std (bool): if true the the plot will show the standard
            deviation curve
        axes (matplotlib.axes.Axes): matplotlib axes used for the plot
        filename (str): if provided a file will be saved with this name
        kwargs (dict): can set figsize
    """

    if not isinstance(psd, PSDS):
        raise PSDSEvalError("The psds data needs to be given as a PSDS object")
    if axes is not None and not isinstance(axes, plt.Axes):
        raise PSDSEvalError("The give axes is not a matplotlib.axes.Axes")

    show = False
    if axes is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
        axes = fig.add_subplot()
        show = True

    axes.vlines(psd.max_efpr, ymin=0, ymax=1.0, linestyles='dashed')
    axes.step(psd.plt.xp, psd.plt.yp, 'b-', label='PSD-ROC', where="post")
    if en_std:
        axes.step(psd.plt.xp,
                  np.maximum(psd.plt.mean - psd.plt.std, 0),
                  c="b", linestyle="--", where="post")
        axes.step(psd.plt.xp, psd.plt.mean + psd.plt.std,
                  c="b", linestyle="--")
    axes.fill_between(psd.plt.xp, y1=psd.plt.yp, y2=0, label="AUC",
                      alpha=0.3, color="tab:blue", linewidth=3, step="post")
    axes.set_xlim([0, psd.max_efpr])
    axes.set_ylim([0, 1.0])
    axes.legend()
    axes.set_ylabel("eTPR")
    axes.set_xlabel(f"eFPR per {psd.duration_unit}")
    axes.set_title(f"PSDS: {psd.value:.5f}\n"
                   f"alpha_st: {psd.alpha_st:.2f}, alpha_ct: "
                   f"{psd.alpha_ct:.2f}, max_efpr: {psd.max_efpr}")
    axes.grid()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_psd_roc(psd, class_names, max_efpr=None,
                           axes=None, filename=None, **kwargs):
    """
    Shows (or saves) the PSD-ROC per class for a given PSDROC object.

    Args:
        psd: PSDROC object as outputted by the psds_eval.psd_roc_curves()
        class_names: the class names generated by the psds_eval object
        max_efpr: the upper limit for the x-axis of the plot. If not given,
                  maximum available value in the plots is used.
        axes: optional matplotlib.pyplot.Axes object to create the plots on
        filename: optional filename to save the figure
        kwargs: additional arguments for pyplot plotting
    """

    if not isinstance(psd, PSDROC):
        raise PSDSEvalError("The psdroc data needs to be a PSDROC object")
    # ignore the artificial world label
    if len(class_names) - 1 != psd.yp.shape[0]:
        raise PSDSEvalError("Num of class names doesn't match the expected")
    if axes is not None and not isinstance(axes, plt.Axes):
        raise PSDSEvalError("The give axes is not a matplotlib.axes.Axes")

    show = False
    if axes is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
        axes = fig.add_subplot()
        show = True

    for i in range(len(class_names) - 1):
        axes.step(psd.xp, psd.yp[i], label=class_names[i], where="post")
    axes.step(psd.xp, psd.mean, lw=2.0, ls="--", label="mean_TPR",
              where="post")

    axes.set_ylim([0, 1.0])
    if max_efpr is None:
        max_efpr = np.max(np.nan_to_num(psd.xp))
    axes.set_xlim([0, kwargs.get("xlim", max_efpr)])
    axes.legend()
    axes.set_xlabel(kwargs.get("xlabel", "(e)FPR"))
    axes.set_ylabel("TPR")
    axes.set_title(kwargs.get("title", "Per-class PSDROC"))
    axes.grid()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
