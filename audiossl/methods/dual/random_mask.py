from fairseq.data.data_utils import compute_mask_indices 
import torch
from torch.nn import functional as F

def get_mask(batch_size,num_patches,mask_ratio,padding_mask=None):
    masks = compute_mask_indices(shape = (batch_size,num_patches),
                        padding_mask=padding_mask,
                        mask_prob=mask_ratio,
                        mask_length=6,
                        mask_type="static",
                        mask_other=None,
                        min_masks=2,
                        no_overlap=False,
                        min_space=0)
    return torch.from_numpy(masks)


def get_mask_v2(batch_size,num_patches,mask_ratio):
    mask_index = []
    for i in range(batch_size):
        mask_index.append((torch.randperm(num_patches) < num_patches*mask_ratio).unsqueeze(0))
    return torch.cat(mask_index)

def get_mask_variable_length(batch_size,num_patches,available_patches,mask_ratio):
    mask_index = []
    available_patches_cpu = available_patches.to("cpu")
    for i in range(batch_size):
        mask_index_ = get_mask_one(num_patches,available_patches_cpu[i],mask_ratio).unsqueeze(0)
        mask_index.append(mask_index_)
    return torch.cat(mask_index)

def get_mask_one(num_patches,available_patches,mask_ratio):
    mask_index_=(torch.randperm(available_patches) < available_patches*mask_ratio)
    mask_index_ = F.pad(mask_index_,(0,num_patches-available_patches),value=1)
    return mask_index_