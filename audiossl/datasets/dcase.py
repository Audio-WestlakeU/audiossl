import yaml
import pandas as pd
import torch
import bisect
import warnings

from collections import OrderedDict
from audiossl.datasets import register_dataset
from .dcase_utils import *

class ConcatDataset(torch.utils.data.dataset.Dataset):
    """
    ref: https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, transform) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.transform = transform
        self.id_0 = 0
        self.id_1 = 0

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if dataset_idx == 0:
            self.id_0 += 1
        else:
            self.id_1 += 1
        # if idx == 255:
        #     print("Dataset index 0:", self.id_0, "index 1:", self.id_1)
        wav, labs, _, filename = self.datasets[dataset_idx][sample_idx]
        audio = self.transform(wav)
        return audio, labs, filename

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

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

def DCASEDataset(conf_file, split, transform=None, target_transform=None, unsup=False):
    with open(conf_file, "r") as f:
        config = yaml.safe_load(f)
    assert target_transform is None, "No label transformation on DCASE data is supported. Mixup is used in training step directly."
    
    # Define label encoder
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    if split == "test":
        # ----------------------------[ Test set ]------------------------------
        # Define test set (real strong)
        test_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        real_test = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            test_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"]
        )
        return ConcatDataset([real_test], transform=transform)

    elif split == "valid":
        # -------------------------[ Validation set ]---------------------------
        # Define validation sets
        # Synthetic dataset 
        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )

        # Weak labeled dataset
        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )
        return ConcatDataset([synth_val, weak_val], transform=transform)
    
    else:
        # --------------------------[ Training set ]----------------------------
        # Define synthetic train set
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True
        )

        # Define real train set
        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )

        # Define unsupervised dataset for training
        if unsup:
            unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
        )
            tot_train_data = [synth_set, weak_set, unlabeled_set]
        else:
            tot_train_data = [synth_set, weak_set]
        train_sampler = ConcatDatasetSampler(tot_train_data, config["training"]["batch_size"], shuffle=True, mode=config["training"]["batch_len_index"], drop_last=True)
        # Define ConcatDataset sampler
        train_sampler_config = {
            "sampler": train_sampler,
            "batch_size": train_sampler.get_bsz(),
        }

        return ConcatDataset(tot_train_data, transform=transform), train_sampler_config


