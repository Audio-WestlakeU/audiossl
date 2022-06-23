
import torchaudio
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity,CentralCrop
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop
from audiossl.transforms.target_transform import MixupSpecLabel
from torchvision import transforms

class FreezingTransform:
    def __init__(self,sr=16000,max_len=9.5):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
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
    def __init__(self,sr=16000,max_len=9.5):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
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
                                [CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                ]
                                )

    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FinetuneEvalTransform:
    def __init__(self,sr=16000,max_len=9.5):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
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
                                [CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                ]
                                )

    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FinetuneTargetTransform:
    def __init__(self,alpha=0.5,n_memory=5000,num_classes=23):
        self.mixup = MixupSpecLabel(alpha,n_memory)
        self.rrc = RandomResizeCrop()
        self.num_classes=num_classes
    def __call__(self,x,y):
        x,y = self.mixup(x,y)
        x = self.rrc(x)
        return x,y