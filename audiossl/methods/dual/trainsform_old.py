
from enum import auto
from assl.transforms.common import Normalize,MinMax,RandomCrop,Identity
from numpy.core.fromnumeric import _around_dispatcher
from torchvision import transforms
import torchaudio
import random
from torch.nn import functional as F
import torch
from torch import nn
import numpy as np
random.seed(1234)
import random_mask
from audio_transformer import get_num_patches
from functools import partial
import matplotlib.pyplot as plt

def plot_spec(x,save_path):
    t = range(0,x.shape[0])
    f = range(0,x.shape[1])
    plt.pcolormesh(x)
    plt.savefig(save_path)
    plt.close()



class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(0)
        return lms.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))

class TimeStrecher:
    def __init__(self,scale=1.0):
        self.scale=scale

        pass
    def __call__(self,x):
        x=F.interpolate(x,scale_factor=(self.scale),mode="linear")
        return x

def energy_scale(xa,xb):
    xa=xa.exp()
    xb=xb.exp()
    return torch.sum(xa)/torch.sum(xb)
def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    len_xa = xa.shape[2]
    len_xb = xb.shape[2]
    if len_xa < len_xb:
        start = np.random.randint(0,len_xb-len_xa)
        scale = energy_scale(xa,xb[:,:,start:start+len_xa])
        scale = 1.0
        xa = alpha*xa +(1.-alpha)*scale*xb[:,:,start:start+len_xa]
        return torch.log(xa + torch.finfo(xa.dtype).eps)
    elif len_xa > len_xb:
        start = np.random.randint(0,len_xa-len_xb)
        scale = energy_scale(xa[:,:,start:start+len_xb],xb)
        scale = 1.0
        xa[:,:,start:start+len_xb] = alpha*xa[:,:,start:start+len_xb] +(1.-alpha)*scale*xb
        return torch.log(xa + torch.finfo(xa.dtype).eps)
    else:
        scale = energy_scale(xa,xb)
        scale = 1.0
        x =  alpha * xa + (1.-alpha)*scale*xb
        return torch.log(x + torch.finfo(x.dtype).eps)


class Mixup(nn.Module):
    """Mixup.

    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio=0.4, n_memory=2000, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]

        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


class MixGaussianNoise():
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp()
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio})'
        return format_string

class MocoTrainTransform:
    def __init__(self,sr=16000,patch_height=64,patch_width=3,mask_ratio=0.75,different_global=True):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = Normalize(reduce_dim=[0,-1])
        normalize = Normalize()
        #normalize = div(100) 
        normalize = MinMax(min=-79.6482,max=50.6842)

        self.different_global=different_global
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mask_ratio = mask_ratio
        self.get_num_patches =  partial(get_num_patches,height=64,patch_height=patch_height,patch_width=patch_width)

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.globalcrop = transforms.Compose(
                                            [RandomCrop(16000*6),
                                             self.mel_feature
                                            ])

        self.global_transform1 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1)),
                                ]
                                )
        self.global_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1)),
                                ]
                                )

    def __call__(self,input):
        crops = []
        lengths = []
        masks = []

        global_len1 = random.uniform(1,6)
        self.globalcrop.transforms[0].size=int(global_len1*16000)

        crop_global1=self.globalcrop(input)

        global_len2 = random.uniform(1,6)
        self.globalcrop.transforms[0].size=int(global_len2*16000)
        crop_global2=self.globalcrop(input) if self.different_global else crop_global1



        mask_global1 = random_mask.get_mask_one(self.get_num_patches(width=int(6*16000)//160+1),
                                                self.get_num_patches(width=int(global_len1*16000)//160+1),
                                                self.mask_ratio)
        mask_global2 = random_mask.get_mask_one(self.get_num_patches(width=int(6*16000)//160+1),
                                                self.get_num_patches(width=int(global_len2*16000)//160+1),
                                                self.mask_ratio)

        crops.append(F.pad(self.global_transform1(crop_global1),
                          (0,int((6*16000)//160-int(global_len1*16000)//160))))
        lengths.append(int(global_len1*16000)//160+1)
        masks.append(mask_global1)
        crops.append(F.pad(self.global_transform2(crop_global2),
                          (0,int((6*16000)//160-int(global_len2*16000)//160))))
        lengths.append(int(global_len2*16000)//160+1)
        masks.append(mask_global2)

        return crops,lengths,masks
    
class DinoTrainTransform:
    def __init__(self,sr=16000,patch_height=64,patch_width=3,mask_ratio=0.75,different_global=True,local_crops_number=6,global_len=(4.,6.),local_len=(0.5,1.),virtual_crop=1.0):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = Normalize(reduce_dim=[0,-1])
        normalize = Normalize()
        #normalize = div(100) 
        normalize = MinMax(min=-79.6482,max=50.6842)

        """
        rrc_global = transforms.RandomResizedCrop((64,101),(0.8,1.0),(1.4,1.8))
        rrc_local = transforms.RandomResizedCrop((64,101),(0.8,1.0),(1.4,1.8))
        rrc_global = Identity()
        rrc_local = Identity()
        global_mask = transforms.Compose(
            [torchaudio.transforms.TimeMasking(6*101//2,iid_masks=True),
                torchaudio.transforms.FrequencyMasking(64//2,iid_masks=True)
            ]
        )
        local_mask = transforms.Compose(
            [torchaudio.transforms.TimeMasking(1*101//2,iid_masks=True),
                torchaudio.transforms.FrequencyMasking(64//2,iid_masks=True)
            ]
        )
        """
        self.different_global=different_global
        self.local_crops_number=local_crops_number
        self.global_len = global_len
        self.local_len = local_len
        self.max_global_len = max(self.global_len)
        self.max_local_len = max(self.local_len)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mask_ratio = mask_ratio
        self.get_num_patches =  partial(get_num_patches,height=64,patch_height=patch_height,patch_width=patch_width)

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.globalcrop = transforms.Compose(
                                            [RandomCrop(16000*6),
                                             self.mel_feature
                                            ])

        self.global_transform1 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )
        self.global_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )

        self.local_transform = transforms.Compose(
                                [
                                RandomCrop(16000*1),
                                self.mel_feature,
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )



    def __call__(self,input):
        crops = []
        lengths = []
        masks = []

        if self.global_len[0] == self.global_len[1] :
            global_len = self.global_len[0]
        else:
            global_len = random.uniform(self.global_len[0],self.global_len[1])

        self.globalcrop.transforms[0].size=int(global_len*16000)

        crop_global1=self.globalcrop(input)

        crop_global2=self.globalcrop(input) if self.different_global else crop_global1



        mask_global1 = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_global_len*16000)//160+1),
                                                self.get_num_patches(width=int(global_len*16000)//160+1),
                                                self.mask_ratio)
        mask_global2 = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_global_len*16000)//160+1),
                                                self.get_num_patches(width=int(global_len*16000)//160+1),
                                                self.mask_ratio)

        crops.append(F.pad(self.global_transform1(crop_global1),
                          (0,int((self.max_global_len*16000)//160-int(global_len*16000)//160))))
        lengths.append(int(global_len*16000)//160+1)
        masks.append(mask_global1)
        crops.append(F.pad(self.global_transform2(crop_global2),
                          (0,int((self.max_global_len*16000)//160-int(global_len*16000)//160))))
        lengths.append(int(global_len*16000)//160+1)
        masks.append(mask_global2)
        for i in range(self.local_crops_number):
            if self.local_len[0] == self.local_len[1] :
                local_len = self.local_len[0]
            else:
                local_len = random.uniform(self.local_len[0],self.local_len[1])
            self.local_transform.transforms[0].size=int(local_len*16000)
            mask_local = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_local_len*16000)//160+1),
                                                  self.get_num_patches(width=int(local_len*16000)//160+1),
                                                  self.mask_ratio)
            crops.append(F.pad(self.local_transform(input),
                          (0,int((self.max_local_len*16000)//160-int(local_len*16000)//160))))
            lengths.append(int(local_len*16000)//160+1)
            masks.append(mask_local)

        return crops,lengths,masks
    
from copy import deepcopy
class MaskDevTrainTransform:
    def __init__(self,sr=16000,patch_height=64,patch_width=3,mask_ratio=0.75,different_global=True,global_crops_number=1,local_crops_number=6,global_len=(4.,6.),local_len=(0.5,1.),virtual_crop=1.0,mask_teacher=0.65):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)

        self.different_global=different_global
        self.global_crops_number=global_crops_number
        self.local_crops_number=local_crops_number
        self.global_len = global_len
        self.local_len = local_len
        self.max_global_len = max(self.global_len)
        self.max_local_len = max(self.local_len)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mask_ratio = mask_ratio
        self.mask_teacher = mask_teacher
        self.get_num_patches =  partial(get_num_patches,height=64,patch_height=patch_height,patch_width=patch_width)


        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.globalcrop = transforms.Compose(
                                            [RandomCrop(16000*6),
                                             self.mel_feature
                                            ])

        self.global_transform1 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1.0),time_scale=(1.0,1.0)),
                                ]
                                )
        self.global_transform1 = Identity()
        self.global_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1.0),time_scale=(1.0,1.0)),
                                ]
                                )


        self.local_transform = transforms.Compose(
                                [
                                RandomCrop(16000*1),
                                self.mel_feature,
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )



    def __call__(self,input):
        crops = []
        lengths = []
        masks_s = []
        masks_t = []
        masks_st = []

        if self.global_len[0] == self.global_len[1] :
            global_len = self.global_len[0]
        else:
            global_len = random.uniform(self.global_len[0],self.global_len[1])

        self.globalcrop.transforms[0].size=int(global_len*16000)

        max_global_patches=self.get_num_patches(width=int(self.max_global_len*16000)//160+1)
        global_patches=self.get_num_patches(width=int(global_len*16000)//160+1)
        length_mask = torch.arange(max_global_patches) < global_patches
        crop_global1=self.globalcrop(input)
        mask_global1_s = random_mask.get_mask(1,
                                            max_global_patches,
                                            self.mask_ratio,
                                            padding_mask= ~length_mask.unsqueeze(0),
                                            ).squeeze(0)
        mask_global1_t = random_mask.get_mask(1,
                                            max_global_patches,
                                            self.mask_teacher,
                                            padding_mask= ~length_mask.unsqueeze(0),
                                            ).squeeze(0)
        """
        mask_global1 = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_global_len*16000)//160+1),
                                                self.get_num_patches(width=int(global_len*16000)//160+1),
                                                0.65)
        """

        crops.append(F.pad(self.global_transform1(crop_global1),
                          (0,int((self.max_global_len*16000)//160-int(global_len*16000)//160))))
        lengths.append(int(global_len*16000)//160+1)
        masks_s.append(mask_global1_s)
        masks_t.append(mask_global1_t)
        if self.mask_teacher > 0:
            while all(~(mask_global1_t & mask_global1_s)): 
                mask_global1_t = random_mask.get_mask(1,
                                                max_global_patches,
                                                self.mask_teacher,
                                                padding_mask= ~length_mask.unsqueeze(0),
                                                ).squeeze(0)
            masks_st.append(  mask_global1_s & mask_global1_t & length_mask)
        else:

            masks_st.append(  mask_global1_s & length_mask)

        if self.global_crops_number == 2 :
            crop_global2 = crop_global1

            crops.append(F.pad(self.global_transform2(crop_global2),
                            (0,int((self.max_global_len*16000)//160-int(global_len*16000)//160))))
            mask_global2 = random_mask.get_mask(1,
                                            max_global_patches,
                                            self.mask_ratio,
                                            padding_mask= ~length_mask.unsqueeze(0),
                                            ).squeeze(0)
            lengths.append(int(global_len*16000)//160+1)
            masks_s.append(mask_global2)
            masks_t.append(mask_global2)
            masks_st.append(mask_global2)

        
        for i in range(self.local_crops_number):
            if self.local_len[0] == self.local_len[1] :
                local_len = self.local_len[0]
            else:
                local_len = random.uniform(self.local_len[0],self.local_len[1])
            self.local_transform.transforms[0].size=int(local_len*16000)
            mask_local = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_local_len*16000)//160+1),
                                                  self.get_num_patches(width=int(local_len*16000)//160+1),
                                                  0)
            crops.append(F.pad(self.local_transform(input),
                          (0,int((self.max_local_len*16000)//160-int(local_len*16000)//160))))
            lengths.append(int(local_len*16000)//160+1)
            masks_s.append(mask_local)
            masks_t.append(mask_local)
            masks_st.append(mask_local)

        return crops,lengths,masks_s,masks_t,masks_st

def compute_src_index(dst_idx,scale,max_src_idx):
    src_idx=1./scale*(dst_idx+0.5)-0.5
    src_idx = torch.clamp(src_idx,0,max_src_idx)
    return src_idx


class MaskAugTrainTransform:
    def __init__(self,sr=16000,patch_height=64,patch_width=3,mask_ratio=0.75,aug_student=True,different_global=True,global_crops_number=1,local_crops_number=6,global_len=(6.,6.),local_len=(0.5,1.),virtual_crop=1.0,mask_teacher=0.65):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)

        self.different_global=different_global
        self.global_crops_number=global_crops_number
        self.local_crops_number=local_crops_number
        self.global_len = global_len
        self.local_len = local_len
        self.max_global_len = max(self.global_len)
        self.max_local_len = max(self.local_len)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mask_ratio = mask_ratio
        self.mask_teacher = mask_teacher
        self.get_num_patches =  partial(get_num_patches,height=64,patch_height=patch_height,patch_width=patch_width)


        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.globalcrop = transforms.Compose(
                                            [RandomCrop(16000*6),
                                             self.mel_feature
                                            ])

        self.global_transform1 = Identity()
        if aug_student:
            self.global_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1.0),time_scale=(1.0,1.0)),
                                TimeStrecher(),
                                ]
                                )

        else:
            self.global_transform2 = Identity()

        self.local_transform = transforms.Compose(
                                [
                                RandomCrop(16000*1),
                                self.mel_feature,
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )



    def __call__(self,input):
        crops = []
        lengths = []
        masks_s = []
        src_idxs = []
        if self.global_len[0] == self.global_len[1] :
            global_len = self.global_len[0]
        else:
            global_len = random.uniform(self.global_len[0],self.global_len[1])

        self.globalcrop.transforms[0].size=int(global_len*16000)

        max_global_patches=self.get_num_patches(width=int(self.max_global_len*16000)//160+1)
        global_patches=self.get_num_patches(width=int(global_len*16000)//160+1)
        length_mask = torch.arange(max_global_patches) < global_patches
        crop_global1=self.globalcrop(input)
        mask_global1_s = random_mask.get_mask(1,
                                            max_global_patches,
                                            self.mask_ratio,
                                            padding_mask= ~length_mask.unsqueeze(0),
                                            ).squeeze(0)

        crops.append(F.pad(self.global_transform1(crop_global1),
                          (0,int((self.max_global_len*16000)//160-int(global_len*16000)//160))))

        crop_global2 = crop_global1

        scale = np.random.uniform(0.6,1.5)
        self.global_transform2.transforms[2].scale=scale
        stretched_crop = self.global_transform2(crop_global2)
        if scale > 1.0:
            stretched_crop = stretched_crop[:,:,:crop_global2.shape[-1]]
            lengths.append((int(global_len*16000)//160+1))
        else:
            lengths.append(int(global_len*16000)//160+1)
        crop_global2_len = stretched_crop.shape[-1]
        global2_patches=self.get_num_patches(width=crop_global2_len)
        length_mask2 = torch.arange(max_global_patches) < global2_patches
        mask_global2_s = random_mask.get_mask(1,
                                            max_global_patches,
                                            self.mask_ratio,
                                            padding_mask= ~length_mask2.unsqueeze(0),
                                            ).squeeze(0)
        mask_idx = torch.arange(max_global_patches)[mask_global2_s]
        src_idx = compute_src_index(torch.arange(global_patches),
                                        scale,
                                        global_patches-1)
        mask_src_idx = torch.round(src_idx[mask_idx]).long()
        mask_src=torch.zeros(max_global_patches).bool()
        mask_src[mask_src_idx] = True

        


        crops.append(F.pad(stretched_crop,
                        (0,int((self.max_global_len*16000)//160 + 1 -crop_global2_len))))
        lengths.append(crop_global2_len)
        masks_s.append(mask_global2_s)
        masks_s.append(mask_global2_s)
        src_idxs.append(src_idx)
        
        for i in range(self.local_crops_number):
            if self.local_len[0] == self.local_len[1] :
                local_len = self.local_len[0]
            else:
                local_len = random.uniform(self.local_len[0],self.local_len[1])
            self.local_transform.transforms[0].size=int(local_len*16000)
            mask_local = random_mask.get_mask_one(self.get_num_patches(width=int(self.max_local_len*16000)//160+1),
                                                  self.get_num_patches(width=int(local_len*16000)//160+1),
                                                  0)
            crops.append(F.pad(self.local_transform(input),
                          (0,int((self.max_local_len*16000)//160-int(local_len*16000)//160))))
            lengths.append(int(local_len*16000)//160+1)
            masks_s.append(mask_local)

        return crops,lengths,masks_s,src_idxs


if __name__ == "__main__":
    import sys
    import datasets
    from torch.utils.data import DataLoader
    data = sys.argv[1]
    transform=MaskAugTrainTransform(different_global=True,local_crops_number=0)
    dset=datasets.LMDBDataset(data,"train",subset=4000,transform=transform)
    dloader= DataLoader(dset,4)
    i = iter(dloader)
    n = next(i)
    rrc1_15 = RandomResizeCrop()
    rrc1_1 = RandomResizeCrop((1,1))
    rrc_g = transforms.RandomResizedCrop((64,601),scale=(0.67,1),ratio=(9.2,9.5))
    rrc_l = transforms.RandomResizedCrop((64,601),scale=(0.05,0.4),ratio=(9,2,9.5))

    import audio_transformer
    ast = audio_transformer.AST(patch_h=64,patch_w=3)
