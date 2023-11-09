import os
import yaml
import pandas as pd
import torch
from .as_strong_utils.as_strong_dict import get_lab_dict
from .dcase_utils import *

class TransformDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, transform):
        super(TransformDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        wav, labs, _, filename = self.dataset[idx]
        audio = self.transform(wav)
        return audio, labs, filename

def ASStrongDataset(as_strong_conf, split, transform=None, target_transform=None):
    assert os.path.exists(as_strong_conf), f"{as_strong_conf} not exist!"
    with open(as_strong_conf, "r") as f:
        config = yaml.safe_load(f)
    assert target_transform is None, "No label transformation on DCASE data is supported. Mixup is used in training step directly."
    
    # Define label encoder
    classes_labels = get_lab_dict(config["data"]["label_dict"])
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
        return TransformDataset(real_test, transform=transform)

    elif split == "valid":
        # -------------------------[ Validation set ]---------------------------
        # Define validation sets
        # Strong dataset 
        strong_df = pd.read_csv(config["data"]["strong_val_tsv"], sep="\t")
        strong_val = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            strong_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
        )


        return TransformDataset(strong_val, transform=transform)
    
    else:
        # --------------------------[ Training set ]----------------------------
        # Define synthetic train set
        strong_df = pd.read_csv(config["data"]["strong_train_tsv"], sep="\t")
        strong_train = StronglyAnnotatedSet(
            config["data"]["strong_folder"],
            strong_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True
        )

        return [TransformDataset(strong_train, transform=transform)]


