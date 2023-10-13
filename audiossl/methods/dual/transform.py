
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity
from torchvision import transforms
import torchaudio
import random
from torch.nn import functional as F
import numpy as np
import random_mask
random.seed(1234)
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop
from audiossl.methods.dual.dual import get_num_patches

    
class DUALTrainTransform:
    def __init__(self,sr=16000,mask_ratio=0.75,seg_len=10):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

        normalize = MinMax(min=-79.6482,max=50.6842)

        self.seg_len = seg_len
        self.mask_ratio=mask_ratio

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.transform = transforms.Compose(
                                            [RandomCrop(16000*self.seg_len),
                                             self.mel_feature
                                            ])
        self.mixup1 = Mixup()
        self.mixup2 = Mixup()

    def __call__(self,input):

        mel=self.transform(input)
        length = mel.shape[-1]
        length = length  - length%16
        mel = mel[:,:,:length]

        #mel_patch = self.mixup1(mel)
        #mel_frame = self.mixup2(mel)
        mel_patch = mel
        mel_frame = mel


        num_patches= get_num_patches(height=64,width=length,patch_height=64,patch_width=16)

        mask = random_mask.get_mask_one(num_patches,num_patches,self.mask_ratio)
        mask = mask.repeat_interleave(4)
        mask_allfalse = mask.clone()
        mask_allfalse[:] = False
        if random.randint(0,1) == 0:
            mask_frame = mask
            mask_patch = mask_allfalse
        else:
            mask_frame = mask_allfalse
            mask_patch = mask
        return mel_frame,mel_patch,length,mask_frame,mask_patch
    


if __name__ == "__main__":
    from model import DUAL
    import sys
    from audiossl import datasets
    from torch.utils.data import DataLoader
    data = sys.argv[1]
    transform=DUALTrainTransform()
    dset=datasets.LMDBDataset(data,"train",subset=4000,transform=transform)
    dloader= DataLoader(dset,4)
    i = iter(dloader)
    n = next(i)
    mel,length,mask_frame,mask_patch = n[0]
    #mask = mask.repeat_interleave(4,dim=1)

    #m = DUAL("small")
    #p_x,f_x,loss=m(mel,mask)



