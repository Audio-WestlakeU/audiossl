from pytorch_lightning import LightningModule
from audiossl.methods.atst.downstream.model import PretrainedEncoderPLModule as ClipEncoder
from audiossl.modules.head import LinearHead
from audiossl.methods.pyramid.downstream.model import PretrainedEncoderPLModule as FrameEncoder
from audiossl.methods.atst.downstream.utils import Metric
import torch


class ClipModel(LightningModule):
    def __init__(self,
                 encoder:ClipEncoder,
                 num_labels:int):
        super().__init__()
        self.encoder= encoder
        self.head = LinearHead(encoder.embed_dim, num_labels,use_norm=True, affine=False)
    def forward(self,batch):
        x,y = self.encoder(batch)
        x = self.head(x)
        return x,y

class FrameModel(LightningModule):
    def __init__(self,
                 encoder:FrameEncoder,
                 nun_labels:int
                 ):
        super().__init__()
        self.encoder= encoder
        self.head = LinearHead(encoder.embed_dim, num_labels,use_norm=True, affine=False)
    def forward(self,batch):
        x,y = self.encoder(batch)
        x = self.head(x)
        return x,y

class EnsembleModel(LightningModule):
    def __init__(self,
                 clip_model:ClipModel,
                 frame_model:FrameModel,
                 multi_label:int):
        super().__init__()
        self.multi_label = multi_label
        if multi_label:
            self.metric = Metric(mode="mAP")
        else:
            self.metric = Metric(mode="ACC") 
        self.clip_model = clip_model
        self.frame_model = frame_model
    def forward(self,batch):
        x1,y = self.clip_model(batch) 
        x2,y = self.frame_model(batch) 
        x = x1 + x2
        if self.multi_label == False and y.dim() > 1:
            y = y.argmax(-1)
        return x,y


    def _cal_metric(self,output,target):
        if self.multi_label:
            self.metric.update(output.sigmoid(),target)
        else:
            self.metric.update(output,target)
    def dump_metric(self):
        metric  = self.metric.compute()
        self.metric.clear()
        print("test_"+self.metric.mode,metric)

from audiossl.lightning.datamodules import (DownstreamDataModule,
                                            get_inmemory_datamodule)
from argparse import ArgumentParser
from audiossl import datasets
from audiossl.methods.pyramid.model import FrameATSTLightningModule

from audiossl.methods.pyramid.downstream.model import FineTuningPLModule as FrameModule
from audiossl.methods.atst.downstream.model import FineTuningPLModule as ClipModule
from audiossl.methods.pyramid.downstream.data import collate_fn
from audiossl.methods.pyramid.downstream.transform import \
    FreezingTransform

from audiossl.lightning.utils import EmbeddingExtractor

def main():
    parser = ArgumentParser("Ensemble")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--clip_ckpt_path", type=str)
    parser.add_argument("--frame_ckpt_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_folds = dataset_info.num_folds

    """load pretrained models"""
    Frame = FrameModule.load_from_checkpoint(args.frame_ckpt_path)
    Clip = Module.load_from_checkpoint(args.clip_ckpt_path)

def get_pretraied_encoder_frame(args):
    # get pretrained encoder
    dict_args = vars(args)

    s = torch.load(args.frame_pretrain_ckpt_path)
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        args.frame_pretrain_ckpt_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder

def get_pretraied_encoder_clip(args):
    # get pretrained encoder
    dict_args = vars(args)

    s = torch.load(args.clip_pretrain_ckpt_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = ATSTLightningModule.load_from_checkpoint(
            args.clip_pretrain_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
    else:
        from audiossl.methods.atst.downstream.utils import \
            load_pretrained_weights
        from audiossl.models.atst.audio_transformer import AST_base, AST_small
        load_args = torch.load(args.clip_pretrain_ckpt_path, map_location="cpu")["args"]
        if load_args.arch=="ast":
            pretrained_encoder = AST_small()
        else:
            pretrained_encoder = AST_base()
        load_pretrained_weights(
            pretrained_encoder, pretrained_weights=args.clip_pretrain_ckpt_path, checkpoint_key="teacher")
    return pretrained_encoder
if __name__ == "__main__":
    #main()
    parser = ArgumentParser("Ensemble")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--clip_ckpt_path", type=str)
    parser.add_argument("--clip_pretrain_ckpt_path", type=str)
    parser.add_argument("--frame_ckpt_path", type=str)
    parser.add_argument("--frame_pretrain_ckpt_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser.add_argument('--n_last_blocks', type=int,  default=1)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_labels = dataset_info.num_labels
    #num_folds = dataset_info.num_folds

    pretrained_encoder = get_pretraied_encoder_frame(args)
    frame_encoder = FrameEncoder(pretrained_encoder,
                                                    6.,
                                                    args.n_last_blocks)
    pretrained_encoder = get_pretraied_encoder_clip(args)
    clip_encoder = ClipEncoder(pretrained_encoder,
                                                    6.,
                                                    args.n_last_blocks)
    """load pretrained models"""
    frame_model = FrameModel(frame_encoder,num_labels)
    frame_model.load_state_dict(torch.load(args.frame_ckpt_path)['state_dict'],strict=False)

    clip_model = ClipModel(clip_encoder,num_labels)
    clip_model.load_state_dict(torch.load(args.clip_ckpt_path)['state_dict'],strict=False)

    dict_args = vars(args)

    """extract embedding"""
    transform = FreezingTransform(n_mels=64)
    data = DownstreamDataModule(**dict_args,
                                fold=None,
                                collate_fn=collate_fn,
                                transforms=[transform]*3,
                                limit_batch_size=min(512,args.batch_size_per_gpu))

    ensemble_model = EnsembleModel(clip_model,frame_model,dataset_info.multi_label)
    ensemble_model = ensemble_model.cuda()
    extracter=EmbeddingExtractor(ensemble_model,nproc=1)
    result = extracter.extract(data.test_dataloader())
    result = [r for r in zip(*result)]
    x_train, y_train = result
    x_train = torch.cat(x_train, dim=0).cuda()
    y_train = torch.cat(y_train, dim=0).cuda()
    if dataset_info.multi_label:
        metric = Metric(mode="mAP")
        metric.update(x_train.sigmoid(),y_train)
    else:
        metric = Metric(mode="ACC") 
        metric.update(x_train,y_train)
    ensemble_model._cal_metric(x_train,y_train)
    ensemble_model.dump_metric()

    