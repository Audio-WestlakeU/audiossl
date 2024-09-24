
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity
from torchvision import transforms
import torchaudio
import random
from torch.nn import functional as F
import torch
random.seed(1234)
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop
from audiossl.models.atst.audio_transformer import get_num_patches
import random_mask

    
class FrameATSTTrainTransform:
    def __init__(self,sr=16000,win_length=1024,aug_tea=True,aug_stu=True,freq_wrap=True,mask_ratio=0.75,mask_nooverlap=False,
                 min_mask_len=2,mask_len=5,mask_type="random",anchor_len=6,patch_h=64,patch_w=4,n_mels=64,**kwargs):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=win_length, n_fft=1024, n_mels=n_mels)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)

        if n_mels==64:
            normalize = MinMax(min=-79.6482,max=50.6842)
        else:
            normalize = MinMax(min=-79.6482,max=50.6842)

        self.anchor_len = anchor_len
        self.max_positive_len = self.anchor_len
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.aug_tea = aug_tea
        self.aug_stu = aug_stu
        self.patch_h=patch_h
        self.patch_w=patch_w
        self.mask_len=mask_len
        self.n_mels=n_mels
        self.mask_nooverlap=mask_nooverlap
        self.min_mask_len=min_mask_len
        self.target_sample_rate = sr
        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )
        self.positivecrop = transforms.Compose(
                                            [RandomCrop(16000*anchor_len),
                                             self.mel_feature
                                            ])

        if self.aug_tea:
            self.positive_transform1 = transforms.Compose(
                                    [
                                    Mixup(),
                                    RandomResizeCrop((1,1.0),time_scale=(1.0,1.0)) if freq_wrap else Identity(),
                                    ]
                                    )
        else:
            self.positive_transform1 = Identity()

        if self.aug_stu:
            self.positive_transform2 = transforms.Compose(
                                [
                                Mixup(),
                                RandomResizeCrop((1,1.0),time_scale=(1.0,1.0)) if freq_wrap else Identity(),
                                ]
                                )
        else:
            self.positive_transform2 = Identity()


    def plot_mel(self, mel_feature, save_mel_img_name):
        # 保存一些频谱图像，确保之前的load，resample，truncate操作大概是对的，和pretrain_dataset_utils用librosa生成的频谱图对比一下
        import matplotlib.pyplot as plt
        import librosa
        plt.figure()
        one_mel_feature = mel_feature.cpu().numpy()[0]
        librosa.display.specshow(one_mel_feature)
        plt.savefig("/20A021/dataset_from_dyl/manifest_ub/segment_random_sample/"+save_mel_img_name+".png")

    def __call__(self, waveform, original_sample_rate):
        crops = []
        lengths = []
        masks = []

        anchor_len = self.anchor_len
        waveform = self.apply_resample_mono(waveform, original_sample_rate)

        self.positivecrop.transforms[0].size=int(anchor_len*16000)

        crop_positive1=self.positivecrop(waveform)

        positive_len = anchor_len
        crop_positive2 = crop_positive1

        num_patches = get_num_patches(self.n_mels,int(anchor_len*16000)//160 + 1,self.patch_h,self.patch_w)

        if self.mask_type == "random":
            mask = random_mask.get_mask_one(num_patches,num_patches,self.mask_ratio)
        elif self.mask_type=="block":
            mask = random_mask.get_mask(1,num_patches,self.mask_ratio,no_overlap=self.mask_nooverlap,min_length=self.mask_len,type="static",other=self.min_mask_len).squeeze(0)
        else:
            mask = random_mask.get_mask(1,num_patches,self.mask_ratio,no_overlap=self.mask_nooverlap,min_length=self.mask_len,type="uniform",other=self.min_mask_len).squeeze(0)

        crops.append(F.pad(self.positive_transform1(crop_positive1),
                          (0,int((self.max_positive_len*16000)//160-int(anchor_len*16000)//160))))
        lengths.append(int(anchor_len*16000)//160+1)
        crops.append(F.pad(self.positive_transform2(crop_positive2),
                          (0,int((self.max_positive_len*16000)//160-int(positive_len*16000)//160))))
        lengths.append(int(positive_len*16000)//160+1)
        masks.extend([mask]*2)
        
        return crops,lengths,masks

    def apply_resample_mono(self, waveform, original_sample_rate):
        if waveform.size(dim=0) > 1: # Convert to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if original_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            #waveform = torchaudio.functional.resample(waveform, orig_freq=original_sample_rate, new_freq=self.target_sample_rate)
        # if resampled_waveform.size(1) != 160000:
        #     print(f"waveform of sr: {original_sample_rate} before and after resample is size:{waveform.size(1)} {resampled_waveform.size(1)}")
        return waveform


