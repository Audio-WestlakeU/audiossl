import torch
import pytorch_lightning as pl
from audiossl.methods.atst.downstream.utils_dcase.comparison_models.frame_atst import FrameAST_base, FrameATSTLightningModule
from audiossl.models.atst.audio_transformer import AST_base
from audiossl.methods.atst.downstream.utils import load_pretrained_weights
from audiossl.methods.atst.downstream.transform import FreezingTransform

class CombineATSTEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.clip_encoder = AST_base(use_cls=True)
        load_pretrained_weights(self.clip_encoder, "/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/dcase_logs/base.ckpt", checkpoint_key="teacher")
        self.n_blocks = 1
        self.frame_encoder = get_frame_atst("/data/home/shaonian/ATST/audiossl/audiossl/methods/atst/downstream/utils_dcase/comparison_models/ckpts/last.ckpt")
        self.embed_dim = self.frame_encoder.embed_dim + self.clip_encoder.embed_dim

    def forward(self, x, length):
        x = x.unsqueeze(1)
        frame_x = self.frame_encoder.get_intermediate_layers(
            x,
            length,
            self.n_blocks,
            scene=False
        )
        clip_x = self.clip_encoder.get_intermediate_layers(
            x,
            length,
            1
        )
        clip_x = [item[:, 1:, :] for item in clip_x]
        clip_x = torch.concat(clip_x, dim=-1)

        x = torch.concat([clip_x, frame_x], dim=-1)
        return x

class CombineATSTPredModule(pl.LightningModule):
    """This module has been modified for frame-level prediction"""

    def __init__(self):
        super().__init__()
        self.transform = FreezingTransform(max_len=10)
        self.encoder = CombineATSTEncoder()
        self.embed_dim = self.encoder.embed_dim
        

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder(x, length)
        return x, y

def get_frame_atst(pretrained_ckpt_path):
    # get pretrained encoder
    print("Loading frame-atst model:")
    s = torch.load(pretrained_ckpt_path)
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        pretrained_ckpt_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder

if __name__ == "__main__":
    
    atst_module = CombineATSTPredModule()
    fake_input = torch.rand(64, 64, 1001)
    lengths = torch.tensor([1001] * 64).float()
    fake_output, _ = atst_module(((fake_input, lengths), 0))
    print(fake_output.shape, fake_input.shape)