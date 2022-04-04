
import torchaudio
from audiossl.transforms.common import Normalize,MinMax,RandomCrop,Identity,CentralCrop
from audiossl.transforms.byol_a import Mixup, RandomResizeCrop
from torchvision import transforms
class FreezingTransferTrainTransform:
    def __init__(self,sr=16000,max_len=12):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr

        rrc_global = Identity()
        rrc_local = Identity()

        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )

        self.global_transform = transforms.Compose(
                                [
                                CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                rrc_global
                                ]
                                )
    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]
