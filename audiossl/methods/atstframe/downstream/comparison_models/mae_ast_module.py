import torch
import torchaudio

import pytorch_lightning as pl
import torch.nn as nn
from audiossl.methods.atstframe.downstream.comparison_models.models.mae_ast import MAE_AST

audio_configs = {
    "feature_type": "fbank", 
    "sample_rate": 16000,
    "max_sample_size": 250000,
    "min_sample_size": 32000,
    "feature_rate": 100,
    "feature_dim": 128,
    "normalize": False, # must be consistent with extractor
    "deltas": False
}

class MAEASTModel(MAE_AST):
    def __init__(self):
        super().__init__()
        self.pad_matrix = torch.zeros(256, 998)
        self.feat_mean = nn.AvgPool2d((2, 1), padding=(0, 0))
    
    def forward(self, source, mask=False, ret_conv=False, output_layer=None):
        res = super().forward(
            source,
            padding_mask=self.pad_matrix.to(source).bool(),
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        feature = torch.concat([feature, feature[:, -1, :].unsqueeze(1)], dim=1)
        feature = self.feat_mean(feature)
        return feature, res["padding_mask"]

class MAEASTPredModule(pl.LightningModule):
    def __init__(self, pertrained_ckpt_path) -> None:
        super(MAEASTPredModule, self).__init__()
        self.encoder = MAEASTModel()
        self.embed_dim = 768
        load_weigts = torch.load(pertrained_ckpt_path)
        state_dicts = load_weigts["model"]
        
        self.encoder.load_state_dict(state_dict=state_dicts, strict=True)
        print("Using layer norm:", self.encoder.encoder.layer_norm_first)
    def forward(self, batch):
        (x, length), y = batch
        x, _ = self.encoder(x)
        return x, y

    def transform(self, wav):
        wav = wav.unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
                waveform=wav,
                sample_frequency=audio_configs["sample_rate"],
                use_energy=False,
                num_mel_bins=audio_configs["feature_dim"],
                frame_shift=10
            )
        feat = feat[:, :audio_configs["feature_dim"]]
        return feat, feat.shape[0]
    
    def finetune_mode(self):
        # self.freeze()
        for p in self.encoder.parameters():
            p.requires_grad = True
        # Unfreeze last tfm block
        # for i, layer in enumerate(self.encoder.encoder.layers):
        #     if i == len(self.encoder.encoder.layers) - 1:
        #         for n, p in layer.named_parameters():
        #             p.requires_grad = True

    def finetune_mannual_train(self):
        # for i, layer in enumerate(self.encoder.encoder.layers):
        #     if i == len(self.encoder.encoder.layers) - 1:
        #         layer.train()
        self.encoder.train()

if __name__ == "__main__":
    module = MAEASTPredModule("/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/random_frame_75_12LayerEncoder.pt")
    model = MAEASTModel()
    print(model)
    fake_input = torch.rand(10, 998, 128)
    fake_output, _ = model(fake_input)
    
    # test_zeros = torch.zeros(1024).reshape(32, -1).unsqueeze(0).unsqueeze(0).float()
    # test_ones = torch.ones(1024).reshape(32, -1).unsqueeze(0).unsqueeze(0).float()
    # test_square = torch.concat([test_zeros, test_ones], dim=-1)
    # print(test_square)
    # unfold_square = model.encoder.unfold(test_square)
    # print(unfold_square)
    print(fake_output.shape)