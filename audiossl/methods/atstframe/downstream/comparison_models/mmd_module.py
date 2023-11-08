from typing import Any
import torch
import nnAudio.features

import pytorch_lightning as pl
import torch.nn as nn

from tqdm import tqdm
from glob import glob
from einops import rearrange
from audiossl.datasets.dcase_utils.datasets import read_audio
from audiossl.methods.atst.downstream.utils_dcase.comparison_models.mmd_model import Config, RuntimeM2D

class MMDModel(RuntimeM2D):
    def __init__(self, weight_file):
        super(MMDModel, self).__init__(weight_file=weight_file)

    def forward(self, x):
        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.cfg.input_size[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        cur_frames = x.shape[-1]
        pad_frames = unit_frames - (cur_frames % unit_frames)

        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))
        embeddings = []
        for i in range(x.shape[-1] // unit_frames):
            emb, *_ = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=0., return_layers=False)
            _, emb = emb[..., :1, :], emb[..., 1:, :]
            emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
            embeddings.append(emb)
        # cut the padding at the end
        x = torch.cat(embeddings, axis=-2)
        pad_emb_frames = int(embeddings[0].shape[-2] * pad_frames / unit_frames)
        # print(2, x.shape, embeddings[0].shape, pad_emb_frames)
        if pad_emb_frames > 0:
            x = x[..., :-(pad_emb_frames + 1), :] # remove padded tail
        # print(3, x.shape)
        return x if len(emb.shape) == 3 else [x_ for x_ in x]
    
class MMDPredModule(pl.LightningModule):
    def __init__(self) -> None:
        super(MMDPredModule, self).__init__()
        self.encoder = MMDModel(weight_file="/home/shaonian/audioset_strong_downstream/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/m2d_vit_base-80x608p16x16-221006-mr6/mmd_ckpt.pth")
        self.embed_dim = 3840
        self.transform = DataTransform()

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x)
        return x, y

    def finetune_mode(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        # self.freeze()
        # # Unfreeze last tfm block
        # for i, layer in enumerate(self.encoder.backbone.blocks):
        #     if i == len(self.encoder.backbone.blocks) - 1:
        #         for n, p in layer.named_parameters():
        #             p.requires_grad = True
        # for n, p in self.encoder.backbone.norm.named_parameters():
        #     p.requires_grad = True

    def finetune_mannual_train(self):
        self.train()
        # for i, layer in enumerate(self.encoder.backbone.blocks):
        #     if i == len(self.encoder.backbone.blocks) - 1:
        #         layer.train()
        # self.encoder.backbone.norm.train()


    def cal_state(self, filename_1, filename_2):
        means, stds = [], []
        filenames = glob(path_1 + "*.wav") + glob(path_2 + "*.wav")
        for file in tqdm(filenames):
            wavs, _, _, _ = read_audio(file, random_channel=False, multisrc=False, pad_to=160000)
            lms = self.encoder.to_feature(wavs)
            means.append(lms.mean())
            stds.append(lms.std())
        print(torch.mean(torch.stack(means)), torch.mean(torch.stack(stds)))

class DataTransform:
    def __init__(self) -> None:
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80,
            fmin=50,
            fmax=8000,
            center=True,
            power=2,
            verbose=False,
        ).to("cpu")
    
    def __call__(self, x) -> Any:
        x = self.to_spec(x)
        lms = (x + torch.finfo().eps).log()
        lms = (lms - (-8.6463)) / (2.6721 + torch.finfo().eps)
        return lms, lms.shape[1]

if __name__ == "__main__": 
    path_1 = "/data/home/shaonian/Datasets/DCASE/dcase2021/dataset/audio/train/synthetic21_train/soundscapes_16k/"
    path_2 = "/data/home/shaonian/Datasets/DCASE/dcase2021/dataset/audio/train/weak_16k/"
    conf_path = "/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/conf/byol_v2.yaml"
    module = MMDPredModule()
    fake_input = torch.rand(10, 160000)
    print("input:", fake_input.shape)
    fake_feature, _ = module.transform(fake_input)
    print("feature:", fake_feature.shape)
    fake_output = module.encoder(fake_feature.unsqueeze(1))
    print("output:", fake_output.shape)
    module.finetune_mode()
    module.finetune_mannual_train()