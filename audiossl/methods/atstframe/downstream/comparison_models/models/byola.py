"""Model definitions.

Reference:
- Y. Koizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “The NTT DCASE2020 challenge task 6 system:
  Automated audio captioning with keywords and sentence length estimation,” DCASE2020 Challenge, Tech. Rep., 2020.
  https://arxiv.org/abs/2007.00225
- D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “Byol for audio: Self-supervised learning
  for general-purpose audio representation,” in IJCNN, Jul 2021.
  https://arxiv.org/abs/2103.06695
"""

import logging
from pathlib import Path
import torch
from torch import nn
import yaml
from easydict import EasyDict

def load_pretrained_weights(model, pathname, model_key='model', strict=True):
    state_dict = torch.load(pathname)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model' in state_dict:
        state_dict = state_dict['model']
    children = sorted([n + '.' for n, _ in model.named_children()])

    # 'model.xxx' -> 'xxx"
    weights = {}
    for k in state_dict:
        weights[k[len(model_key)+1:] if k.startswith(model_key+'.') else k] = state_dict[k]
    state_dict = weights

    # model's parameter only
    def find_model_prm(k):
        for name in children:
            if name in k: # ex) "conv_block1" in "model.conv_block1.conv1.weight"
                return k
        return None

    weights = {}
    for k in state_dict:
        if find_model_prm(k) is None: continue
        weights[k] = state_dict[k]

    logging.info(f' using network pretrained weight: {Path(pathname).name}')
    print(list(weights.keys()))
    logging.info(str(model.load_state_dict(weights, strict=strict)))
    return sorted(list(weights.keys()))


def mean_max_pooling(frame_embeddings):
    assert len(frame_embeddings.shape) == 3 # Batch,Time,Dimension
    (x1, _) = torch.max(frame_embeddings, dim=1)
    x2 = torch.mean(frame_embeddings, dim=1)
    x = x1 + x2
    return x


class AudioNTT2022Encoder(nn.Module):
    """General Audio Feature Encoder Network"""

    def __init__(self, n_mels, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True):
        super().__init__()
        convs = [
            nn.Conv2d(1, base_d, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_d),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ]
        for c in range(1, conv_layers):
            convs.extend([
                nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
                nn.BatchNorm2d(base_d),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            ])
        self.features = nn.Sequential(*convs)
        self.conv_d = base_d * (n_mels//(2**conv_layers))
        self.fc = nn.Sequential(
            nn.Linear(self.conv_d, mlp_hidden_d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden_d, d - self.conv_d),
            nn.ReLU(),
        )
        self.stack = stack

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x_fc = self.fc(x)
        x = torch.hstack([x.transpose(1,2), x_fc.transpose(1,2)]).transpose(1,2) if self.stack else x_fc
        return x


class AudioNTT2022(AudioNTT2022Encoder):
    def __init__(self, n_mels, d=3072, mlp_hidden_d=2048):
        super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)
        self.embed_dim = d
    def forward(self, x):
        x = super().forward(x)
        x = mean_max_pooling(x)
        return x




if __name__ == "__main__":
    from byola_module import load_yaml_config
    from nnAudio.features import MelSpectrogram
    from byola_module import PrecomputedNorm
    print("load configs")
    cfg = load_yaml_config("../../conf/byol_v2.yaml")
    print(cfg)

    wav = torch.rand(1, 160000)
    stats = [-9.660292, 4.7219563]
    print("faeture extraction")
    to_melspec = MelSpectrogram(
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
    print("construct model")
    model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
    print(model)
    print("loading weights")
    load_pretrained_weights(model, './ckpts/AudioNTT2022-BYOLA-64x96d2048.pth')
    print("normalization")
    normalizer = PrecomputedNorm(stats)
    lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())
    print(lms.shape)
    features = model(lms.unsqueeze(0))
    print(features.shape)