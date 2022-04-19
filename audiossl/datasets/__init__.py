
from pathlib import Path
import os

from .registry import register_dataset,list_all_datasets,get_dataset

from .lmdb import LMDBDataset
from .byol_a import Nsynth,Urbansound8k
from .voxceleb1 import SpeakerClassifiDataset
from .librispeech import LibriSpeechDataset
from .iemocap import IEMOCAPDataset

@register_dataset("voxceleb1",multi_label=False,num_labels=1251,num_folds=1)
def create_voxceleb1(data_path,split,transform,target_transform):
    if split == "valid":
        split = "dev"
    return SpeakerClassifiDataset(split,
                                Path(data_path),
                                Path(os.path.join(data_path,"iden_split.txt")),
                                transform=transform,
                                target_transform=target_transform)
@register_dataset("us8k",multi_label=False,num_labels=10,num_folds=10)
def create_us8k(data_path,split,fold,transform,target_transform):
    return Urbansound8k(data_path,split=split,valid_fold=fold,transform=transform)
    
@register_dataset("nsynth",multi_label=False,num_labels=11,num_folds=1)
def create_nsynth(data_path,split,transform,target_transform):
    return Nsynth(data_path,split=split,transform=transform)

@register_dataset("spcv2",multi_label=False,num_labels=35,num_folds=1)
def create_spcv2(data_path,split,transform,target_transform):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform) 

@register_dataset("audioset_b",multi_label=True,num_labels=527,num_folds=1)
def create_spcv2(data_path,split,transform,target_transform):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform) 

@register_dataset("audioset",multi_label=True,num_labels=527,num_folds=1)
def create_spcv2(data_path,split,transform,target_transform):
    if split == "test":
        split = "eval"
    return LMDBDataset(data_path,split=split,transform=transform,target_transform=target_transform) 



__all__ = ['LMDBDataset',
           'Nsynth',
           'Urbansound8k',
           'LibriSpeechDataset',
           'SpeakerClassifiDataset',
           'IEMOCAPDataset']