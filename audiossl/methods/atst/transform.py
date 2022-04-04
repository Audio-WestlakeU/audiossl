
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity
from torchvision import transforms
import torchaudio
import random
from torch.nn import functional as F
import numpy as np
random.seed(1234)
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop

    
class ATSTTrainTransform:
    def __init__(self,sr=16000,mask_ratio=0.75,different_positive=True,anchor_len=(6.,6.),positive_len=(6.,6.),virtual_crop=1.5):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

        normalize = MinMax(min=-79.6482,max=50.6842)

        self.different_positive=different_positive
        self.anchor_len = anchor_len
        self.positive_len = positive_len
        self.max_positive_len = max(self.positive_len + self.anchor_len)

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.positivecrop = transforms.Compose(
                                            [RandomCrop(16000*6),
                                             self.mel_feature
                                            ])

        self.positive_transform1 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )
        self.positive_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,virtual_crop)),
                                ]
                                )



    def __call__(self,input):
        crops = []
        lengths = []

        anchor_len = random.uniform(self.anchor_len[0],self.anchor_len[1])

        self.positivecrop.transforms[0].size=int(anchor_len*16000)

        crop_positive1=self.positivecrop(input)

        if self.different_positive:
            positive_len = random.uniform(self.positive_len[0],self.positive_len[1])
            self.positivecrop.transforms[0].size=int(positive_len*16000)
            crop_positive2=self.positivecrop(input)
        else:
            positive_len = anchor_len
            crop_positive2 = crop_positive1

        crops.append(F.pad(self.positive_transform1(crop_positive1),
                          (0,int((self.max_positive_len*16000)//160-int(anchor_len*16000)//160))))
        lengths.append(int(anchor_len*16000)//160+1)
        crops.append(F.pad(self.positive_transform2(crop_positive2),
                          (0,int((self.max_positive_len*16000)//160-int(positive_len*16000)//160))))
        lengths.append(int(positive_len*16000)//160+1)
        return crops,lengths
    

