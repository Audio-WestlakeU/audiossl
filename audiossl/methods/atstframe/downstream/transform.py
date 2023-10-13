
import torchaudio
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity,CentralCrop
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop
from audiossl.transforms.target_transform import MixupSpecLabel,MixupSpecLabelAudioset
from torchvision import transforms
import torch
import numpy as np

class FreezingTransform:
    def __init__(self,sr=16000,max_len=12,n_mels=64,win_length=1024):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=win_length, n_fft=1024, n_mels=n_mels)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr


        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )

        self.global_transform = transforms.Compose(
                                [
                                CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                ]
                                )
    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FinetuneTrainTransform:
    def __init__(self,sr=16000,max_len=12,n_mels=64,pad=False,is_roll_mag=False):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=n_mels)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr
        self.len=len
        if is_roll_mag:
            roll_mag=RollMag()

            self.mel_feature = transforms.Compose(
                                    [roll_mag,
                                     melspec_t,
                                    to_db,
                                    normalize]
                                    )
        else:
            self.mel_feature = transforms.Compose(
                                    [melspec_t,
                                    to_db,
                                    normalize]
                                    )

        self.global_transform = transforms.Compose(
                                [CentralCrop(int(sr*max_len),pad=pad),
                                self.mel_feature,
                                ]
                                )

    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FinetuneEvalTransform:
    def __init__(self,sr=16000,max_len=12,n_mels=64,pad=False):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=n_mels)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr
        self.len=len

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )

        self.global_transform = transforms.Compose(
                                [CentralCrop(int(sr*max_len),pad=pad),
                                self.mel_feature,
                                ]
                                )

    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FinetuneTargetTransform:
    def __init__(self,alpha=0.5,n_memory=5000,num_classes=23):
        self.mixup = MixupSpecLabel(alpha=alpha,n_memory=n_memory)
        #self.rrc = RandomResizeCrop((1,1.0),time_scale=(1.0,1.0))
        self.rrc = RandomResizeCrop()
        self.num_classes=num_classes
    def __call__(self,x,y):
        x,y = self.mixup(x,y)
        x = self.rrc(x)
        return x,y

def roll_mag_aug(waveform):
    waveform=waveform.numpy()
    idx=np.random.randint(waveform.shape[-1])
    rolled_waveform=np.roll(waveform,idx,axis=-1)
    mag = np.random.beta(10, 10) + 0.5
    return torch.Tensor(rolled_waveform*mag)
class RollMag:
    def __init__(self):
        pass
    def __call__(self,x):
        return roll_mag_aug(x)
class MixupSpecLabelAudioset:
    def __init__(self,dataset,mixup_ratio=0.5,alpha=10,n_memory=5000,num_classes=100):
        self.dataset = dataset
        self.mixup_ratio = mixup_ratio
        self.alpha = alpha
        self.memory_bank = []
        self.n=n_memory
        self.num_classes=num_classes
    def __call__(self,x,y):

        #rank = dist.get_rank()
        #print("rank {}".format(rank),"id {}".format(get_worker_info()),"seed {}".format(np.random.get_state()[1][0]))

        def convert_label(y):
            """convert label to one hot vector"""
            if isinstance(y,int) or (isinstance(y,torch.Tensor) and (len(y.shape)==0 or y.shape[-1]==1) ):
                if isinstance(y,int):
                    y=torch.nn.functional.one_hot(torch.tensor(y),num_classes=self.num_classes)
                else:
                    y=torch.nn.functional.one_hot(y.to(torch.int64),num_classes=self.num_classes)
            return y
        y = convert_label(y)

        if  np.random.random()<self.mixup_ratio:
            l = np.random.beta(self.alpha,self.alpha,1)
            index = np.random.randint(len(self.dataset))
            (x_,_),y_ = self.dataset[index]
            y_=convert_label(y_)
            if x.shape[-1] == x_.shape[-1]:
                x_mix = x*l + x_*(1-l)
            elif x.shape[-1] > x_.shape[-1]:
                start = np.random.randint(0,x.shape[-1] - x_.shape[-1])
                x_mix = x.clone()

                x_mix[:,:,start:start+x_.shape[-1]] = x[:,:,start:start+x_.shape[-1]]*l + x_*(1-l)
            else:
                start = np.random.randint(0,x_.shape[-1] - x.shape[-1])

                x_mix= x*l + x_[:,:,start:start+x.shape[-1]]*(1-l)
            y_mix = y*l +y_ * (1-l)
        else:
            x_mix=x
            y_mix=y

        return x_mix.to(torch.float),y_mix.to(torch.float)
from torchaudio.transforms import FrequencyMasking,TimeMasking
class FinetuneTargetTransformAudioset:
    def __init__(self,dataset,is_mask_aug,is_rrc,alpha=10,mixup_ratio=0.5,num_classes=23):
        self.mixup = MixupSpecLabelAudioset(dataset,mixup_ratio=mixup_ratio,alpha=alpha)
        self.is_mask_aug = is_mask_aug
        self.is_rrc = is_rrc
        #self.rrc = RandomResizeCrop((1,1.0),time_scale=(1.0,1.0))
        if self.is_rrc:
            self.rrc = RandomResizeCrop()
        if self.is_mask_aug:
            self.freq_mask = FrequencyMasking(freq_mask_param=20)
            self.time_mask = TimeMasking(time_mask_param=50)
        self.num_classes=num_classes
    def __call__(self,x,y):
        x,y = self.mixup(x,y)
        if self.is_mask_aug:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        if self.is_rrc:
            x = self.rrc(x)
        return x,y
