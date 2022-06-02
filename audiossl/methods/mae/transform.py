
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity
from torchvision import transforms
import torchaudio
import random
from torch.nn import functional as F
import numpy as np
random.seed(1234)
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop

    
class MAETrainTransform:
    def __init__(self,sr=16000,mask_ratio=0.75,seg_len=6.):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

        normalize = MinMax(min=-79.6482,max=50.6842)

        self.seg_len = seg_len

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.transform = transforms.Compose(
                                            [RandomCrop(16000*self.seg_len),
                                             self.mel_feature
                                            ])

    def __call__(self,input):



        mel=self.transform(input)
        length = mel.shape[-1]

        return mel,length
    

