
from pathlib import Path
import os

from .registry import register_dataset,list_all_datasets,get_dataset

from .lmdb import LMDBDataset
from .byol_a import Nsynth,Urbansound8k
from .voxceleb1 import SpeakerClassifiDataset
from .librispeech import LibriSpeechDataset
from .iemocap import IEMOCAPDataset
from .dcase import DCASEDataset
from .as_strong import ASStrongDataset


@register_dataset("voxceleb1",multi_label=False,num_labels=1251,num_folds=1)
def create_voxceleb1(data_path,split,transform,target_transform,return_key=False):
    if split == "valid":
        split = "dev"
    return SpeakerClassifiDataset(split,
                                Path(data_path),
                                Path(os.path.join(data_path,"iden_split.txt")),
                                transform=transform,
                                target_transform=target_transform,
                                return_key=return_key)
@register_dataset("us8k",multi_label=False,num_labels=10,num_folds=10)
def create_us8k(data_path,split,fold,transform,target_transform,return_key=False):
    return Urbansound8k(data_path,split=split,valid_fold=fold,transform=transform,return_key=return_key)
    
@register_dataset("nsynth",multi_label=False,num_labels=11,num_folds=1)
def create_nsynth(data_path,split,transform,target_transform,return_key=False):
    return Nsynth(data_path,split=split,transform=transform,return_key=return_key)

@register_dataset("spcv2",multi_label=False,num_labels=35,num_folds=1)
def create_spcv2(data_path,split,transform,target_transform,return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform,return_key=return_key) 

@register_dataset("fsd50k",multi_label=True,num_labels=200,num_folds=1)
def create_fsd50k(data_path,split,transform,target_transform,return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform,return_key=return_key) 

@register_dataset("audioset_b",multi_label=True,num_labels=527,num_folds=1)
def create_audioset_b(data_path,split,transform,target_transform,return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform,return_key=return_key) 

@register_dataset("audioset",multi_label=True,num_labels=527,num_folds=1)
def create_audioset(data_path,split,transform,target_transform,return_key=False):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform,return_key=return_key) 

# [DCASE MARK] add a register for dcase dataset
@register_dataset("dcase", multi_label=True, num_labels=10, num_folds=1)
def create_dcase(config_path, split, transform=None, target_transform=None, unsup=False,return_key=False):
    assert split in ["train", "valid", "test"], "Dataset type: {} is not supported.".format(split)
    return DCASEDataset(config_path, split, transform=transform, target_transform=None, unsup=unsup)

@register_dataset("as_strong", multi_label=True, num_labels=407, num_folds=1)
def create_dcase(as_strong_conf, split, transform=None, target_transform=None):
    assert split in ["train", "valid", "test"], "Dataset type: {} is not supported.".format(split)
    return ASStrongDataset(as_strong_conf, split, transform=transform, target_transform=None)

__all__ = ['LMDBDataset',
           'Nsynth',
           'Urbansound8k',
           'LibriSpeechDataset',
           'SpeakerClassifiDataset',
           'IEMOCAPDataset']