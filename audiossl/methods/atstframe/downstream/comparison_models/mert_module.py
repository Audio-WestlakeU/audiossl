import pytorch_lightning as pl
import torch
from torchvision import transforms
from transformers import AutoModel,Wav2Vec2FeatureExtractor
import torch.nn.functional as F
import torch.nn as nn

from audiossl.transforms.common import CentralCrop


class MertTransform:
    def __init__(self, url, max_len=10):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(url, local_files_only=True, trust_remote_code=True)
        self.sr = self.processor.sampling_rate
        self.global_transform = transforms.Compose(
                                [
                                CentralCrop(int(self.sr * max_len), pad=False),
                                ]
                                )

    def __call__(self, x):
        x = self.global_transform(x)
        output = self.processor(x, sampling_rate=self.sr, return_tensors="pt")
        input_values = output['input_values']
        return input_values[0], input_values.shape[-1]


class MertPredModule(pl.LightningModule):
    def __init__(self, pretrained_model_path="/20A021/compare_with/mert/pretrained_model/MERT-v0",
                 freeze_all=False, use_last=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_path, local_files_only=True, trust_remote_code=True)
        self.embed_dim = 768
        self.transform = MertTransform(url=pretrained_model_path)
        self.freeze_all = freeze_all
        self.use_last = use_last
        self.layer_weights = torch.nn.parameter.Parameter(data=torch.ones(13), requires_grad=True)
        pool_size = 2 if 'v0' in pretrained_model_path else 3  # v0是50hz，需要2帧合并一帧。v1是75hz，需要3帧并一帧
        self.feat_mean = nn.AvgPool1d(pool_size, pool_size)

    def forward(self, batch):
        (x, length), y = batch  # x: 64, 240000   length: [64] value都是240000  y: 64, 7, 375
        x = self.encoder(x, output_hidden_states=True)
        if self.use_last:
            h = x["last_hidden_state"]
            pad_width = (0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        else:
            h = x["hidden_states"]
            h = torch.stack(h, dim=3)
            pad_width = (0, 0, 0, 0, 0, 1)
            h = F.pad(h, pad_width, mode='reflect')
        if not self.use_last:
            weights = torch.softmax(self.layer_weights, dim=0)
            h = torch.matmul(h, weights)
        h = h.transpose(1,2)  # [64, 375, 768] ---> [64, 768, 375]
        pooled_h = self.feat_mean(h)  # [64, 768, 125]
        pooled_h = pooled_h.transpose(1, 2)  # 恢复成[64,125, 768]
        return pooled_h, y  # torch.Size([64, 125, 768]) (64, 7,125)

    def finetune_mode(self):
        if self.freeze_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder.feature_extractor._freeze_parameters()

    def finetune_mannual_train(self):
        if self.freeze_all:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder.feature_extractor._freeze_parameters()
