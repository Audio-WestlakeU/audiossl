import yaml
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
import pytorch_lightning as pl
import torch.nn as nn
from audiossl.methods.atst.downstream.utils_dcase.comparison_models.byola import AudioNTT2022Encoder, load_pretrained_weights
from nnAudio.features import MelSpectrogram

    
class BYOLAPredModule(pl.LightningModule):
    def __init__(self, pretrained_ckpt_path, path_to_config="/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/conf/byol_v2.yaml") -> None:
        super(BYOLAPredModule, self).__init__()
        cfg = load_yaml_config(path_to_config)
        self.encoder = AudioNTT2022Encoder(n_mels=cfg.n_mels, d=cfg.feature_d)
        load_pretrained_weights(self.encoder, pretrained_ckpt_path)
        self.embed_dim = 3072
        self.transform = DataTransform(cfg)
        self.unfreeze_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x)
        return x, y
    
    def finetune_mode(self):
        self.freeze()
        
        # Unfreeze last tfm block
        for i in self.unfreeze_ids:
            layer = getattr(self.encoder.features, f"{i}")
            for n, p in layer.named_parameters():
                p.requires_grad = True

        # Unfreeze last norm layer
        for n, p in self.encoder.fc.named_parameters():
            p.requires_grad = True


    def finetune_mannual_train(self):
        for i in self.unfreeze_ids:
            layer = getattr(self.encoder.features, f"{i}")
            layer.train()
        self.encoder.fc.train()


def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file()
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg

class PrecomputedNorm(nn.Module):
    """Normalization using Pre-computed Mean/Std.
    Args:
        stats: Precomputed (mean, std).
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, stats):
        super().__init__()
        self.mean, self.std = stats

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return ((X - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
        return format_string
    
class DataTransform:
    def __init__(self, cfg) -> None:
        self.mel_feat = MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
            )
        
        stat = [-6.596029, 3.5494373]
        self.normalizer = PrecomputedNorm(stat)
        
    def __call__(self, wav):
        output = self.normalizer((self.mel_feat(wav) + torch.finfo(torch.float).eps).log())
        return output,output.shape[-1]

def calc_norm_stats(path_1, path_2, path_to_config):
    from glob import glob
    from audiossl.datasets.dcase_utils.datasets import read_audio
    from tqdm import tqdm
    cfg = load_yaml_config(path_to_config)
    to_spec = MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        center=True,
        power=2,
        verbose=False,
    )
    # Calculate normalization statistics from the training dataset.
    filenames = glob(path_1 + "*.wav") + glob(path_2 + "*.wav")
    # filenames = filenames[:100]
    n_stats = min(20000, len(filenames))
    X = []
    for file in tqdm(filenames):
        wavs, _, _, _ = read_audio(file, random_channel=False, multisrc=False, pad_to=160000)
        lms_batch = (to_spec(wavs) + torch.finfo().eps).log().unsqueeze(1)
        X.extend([x for x in lms_batch.detach().cpu().numpy()])
        if len(X) >= n_stats: break
    X = np.stack(X)
    norm_stats = np.array([X.mean(), X.std()])
    print(norm_stats)
    return norm_stats


if __name__ == "__main__":
    # pretrained_module = BYOLAPredModule("/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/AudioNTT2022-BYOLA-64x96d2048.pth")
    # fake_data = torch.rand([10, 160000])
    # fake_fb, length = pretrained_module.transform(fake_data)
    # fake_output, _ = pretrained_module(((fake_fb, 0), 0))
    # print(fake_fb.shape, fake_output.shape)
    # print(length)
    path_1 = "/data/home/shaonian/Datasets/DCASE/dcase2021/dataset/audio/train/synthetic21_train/soundscapes_16k/"
    path_2 = "/data/home/shaonian/Datasets/DCASE/dcase2021/dataset/audio/train/weak_16k/"
    conf_path = "/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/conf/byol_v2.yaml"
    calc_norm_stats(path_1, path_2, conf_path)