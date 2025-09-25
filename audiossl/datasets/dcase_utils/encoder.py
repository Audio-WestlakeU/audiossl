import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from dcase_util.data import DecisionEncoder
from pathlib import Path

class ManyHotEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """

    def __init__(
        self, labels, audio_len, frame_len, frame_hop, net_pooling=1, fs=16000
    ):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling
        n_frames = self.audio_len * self.fs
        # self.n_frames = int(
        #     int(((n_frames - self.frame_len) / self.frame_hop)) / self.net_pooling
        # )
        self.n_frames = int(int((n_frames / self.frame_hop)) / self.net_pooling)

    def encode_weak(self, labels):
        """ Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if type(labels) is str:
            if labels == "empty":
                y = np.zeros(len(self.labels)) - 1
                return y
            else:
                labels = labels.split(",")
        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1
        return y

    def _time_to_frame(self, time):
        samples = time * self.fs
        frame = (samples) / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        frame = frame * self.net_pooling / (self.fs / self.frame_hop)
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert any(
            [x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]]
        )

        samples_len = self.n_frames
        if type(label_df) is str:
            if label_df == "empty":
                y = np.zeros((samples_len, len(self.labels))) - 1
                return y
        y = np.zeros((samples_len, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = int(self._time_to_frame(row["onset"]))
                        offset = int(np.ceil(self._time_to_frame(row["offset"])))
                        y[
                            onset:offset, i
                        ] = 1  # means offset not included (hypothesis of overlapping frames, so ok)

        elif type(label_df) in [
            pd.Series,
            list,
            np.ndarray,
        ]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(
                    label_df.index
                ):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(self._time_to_frame(label_df["onset"]))
                        offset = int(np.ceil(self._time_to_frame(label_df["offset"])))
                        y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label != "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] != "":
                        i = self.labels.index(event_label[0])
                        onset = int(self._time_to_frame(event_label[1]))
                        offset = int(np.ceil(self._time_to_frame(event_label[2])))
                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError(
                        "cannot encode strong, type mismatch: {}".format(
                            type(event_label)
                        )
                    )

        else:
            raise NotImplementedError(
                "To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                "columns, or it is a list or pandas Series of event labels, "
                "type given: {}".format(type(label_df))
            )
        return y

    def decode_weak(self, labels):
        """ Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels):
        """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append(
                    [
                        self.labels[i],
                        self._frame_to_time(row[0]),
                        self._frame_to_time(row[1]),
                    ]
                )
        return result_labels

    def gpu_decode_strong(self, labels, thds, filenames, output_type="pandas"):
        # decode strong predictions by gpu
        # labels: [Thds, Bsz, Cls, Time]
        assert len(thds) == labels.shape[0], "Label thresholds and provided thresholds not match"
        # get onset/offset
        padded_labels = F.pad(labels, (1, 1, 0, 0), value=0, mode="constant")
        onset_mat = labels - padded_labels[:, :, :, :-2]
        offset_mat = labels - padded_labels[:, :, :, 2:]
        onset_mat = onset_mat.reshape(-1, onset_mat.shape[-1]) # [Thds * Bsz * Cls, Time]
        offset_mat = offset_mat.reshape(-1, offset_mat.shape[-1])
        # Mask silent frames
        onset_offset_mask = onset_mat.abs().sum(-1) > 0
        onset_mat = onset_mat[onset_offset_mask]
        offset_mat = offset_mat[onset_offset_mask]
        onset_index = torch.argwhere(onset_mat == 1)
        offset_index = torch.argwhere(offset_mat == 1)
        assert torch.equal(onset_index[:, 0], offset_index[:, 0]), "onset/offset detect mismatch, need debug"
        if output_type == "pandas":
            # Create thd-filename-cls list (according to reshape order)
            thd_filename_cls = []
            str_filenames = [Path(x).stem + ".wav" for x in filenames]
            str_thds = [str(thd) for thd in thds]
            for thd in str_thds:
                for filename in str_filenames:
                    for c in self.labels:
                        thd_filename_cls.append(thd + " " + filename + " " + c)
            thd_filename_cls = np.array(thd_filename_cls)
            thd_filename_cls = thd_filename_cls[onset_offset_mask.cpu().numpy()]
            event_index = onset_index[:, 0]
            onset_timestamps = onset_index[:, 1] * self.net_pooling / (self.fs / self.frame_hop)
            offset_timestamps = (offset_index[:, 1] + 1) * self.net_pooling / (self.fs / self.frame_hop)     # plus one to meet original ManyHotEncoder setups
            thd_filename_cls = thd_filename_cls[event_index.cpu().numpy()]
            thds_all = [float(x.split(" ")[0]) for x in thd_filename_cls]
            #print("thds_all:",thds_all)
            #print("========================")
            filename_all = [x.split(" ")[1] for x in thd_filename_cls]
            cls_all = [x.split(" ")[2] for x in thd_filename_cls]
            onset_timestamps = onset_timestamps.cpu().numpy()
            offset_timestamps = offset_timestamps.cpu().numpy()
            # Creat pandas dataframe
            df = pd.DataFrame(
                {
                    "thd": thds_all,
                    "filename": filename_all,
                    "event_label": cls_all,
                    "onset": onset_timestamps,
                    "offset": offset_timestamps,
                }
            )
            
            return_dict = {k: g.iloc[:, 1:] for k, g in df.groupby("thd")}
            #print(df.groupby(["thd"]))
            #print("retrun keys",return_dict.keys())
            #print("=======================")
            # Recheck thresholds
            empty_thds = [thd for thd in thds if thd not in return_dict.keys()]
            for t in empty_thds:
                return_dict[t] = pd.DataFrame([], columns=["filename", "event_label", "onset", "offset"])
            return return_dict
        elif output_type == "tensor":
            raise NotImplementedError

        pass

    def state_dict(self):
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        audio_len = state_dict["audio_len"]
        frame_len = state_dict["frame_len"]
        frame_hop = state_dict["frame_hop"]
        net_pooling = state_dict["net_pooling"]
        fs = state_dict["fs"]
        return cls(labels, audio_len, frame_len, frame_hop, net_pooling, fs)
