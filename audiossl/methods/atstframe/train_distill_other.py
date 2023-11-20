import torch
from audiossl.methods.atstframe.model import FrameATSTLightningModule
from audiossl.methods.atst.downstream.train_freeze import get_pretraied_encoder
from argparse import ArgumentParser
from audiossl.methods.atstframe.module_distill_other import DistillLightningModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from audiossl.lightning.datamodules import DownstreamDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from audiossl.methods.atstframe.downstream.data import collate_fn
from audiossl.methods.atstframe.module_distill_other import DistillATSTTrainTransform,DistillATSTEvalTransform,DistllATSTTargetTransform
from audiossl.methods.atstframe.downstream.transform import \
     FinetuneTargetTransform
from audiossl import datasets
import os



def get_pretraied_encoder_frame(args):
    # get pretrained encoder
    dict_args = vars(args)

    s = torch.load(args.pretrained_ckpt_path_frame,map_location="cpu")
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        args.pretrained_ckpt_path_frame)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    return pretrained_encoder

#def main():


if __name__ == "__main__":
    parser = ArgumentParser("LinearClassifier")
    from audiossl.utils.common import bool_flag
    #parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--pretrained_ckpt_path_clip", type=str)
    parser.add_argument("--pretrained_ckpt_path_frame", type=str)
    parser.add_argument("--alpha", type=float,default=0.5)
    parser.add_argument("--mixup_training", type=bool_flag,default=False)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = DistillLightningModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")

    train_transform=DistillATSTTrainTransform()
    eval_transform = DistillATSTEvalTransform()
    if args.mixup_training:
        target_transform = FinetuneTargetTransform(alpha=args.alpha,
                                                   num_classes=datasets.get_dataset(args.dataset_name).num_labels)
    else:
        target_transform = None

    data = DownstreamDataModule(**dict_args,
                            batch_size=args.batch_size_per_gpu,
                            fold=None,
                            collate_fn=collate_fn,
                            transforms=[train_transform,eval_transform,eval_transform],
                            target_transforms=[target_transform,None,None])
    #load model
    cls_s = torch.load(args.pretrained_ckpt_path_clip,map_location="cpu")
    frame_encoder = get_pretraied_encoder_frame(args)

    model = DistillLightningModule(
                                    multi_label=data.multi_label,
                                    nclasses=datasets.get_dataset(args.dataset_name).num_labels,
                                    niter_per_epoch=len(data.dataset_train)//args.batch_size_per_gpu//args.nproc+1,
                                    **dict_args)                            
    print(len(data.dataset_train))
    state_dict_cls = cls_s["state_dict"]
    state_dict_cls = {k.replace("encoder.encoder","encoder"):v for k,v in state_dict_cls.items()}
    r1=model.model.teacher.load_state_dict(state_dict_cls,strict=False)
    r2=model.model.student.load_state_dict(frame_encoder.state_dict(),strict=False)
    for n,p in model.model.teacher.named_parameters():
        p.requires_grad = False
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            gpus=args.nproc,
                            gradient_clip_val=3.0,
                            max_epochs=args.max_epochs,
                            check_val_every_n_epoch=1,
                            logger=[logger_tb],
                            callbacks=[ModelCheckpoint(dirpath=args.save_path,
                                                       every_n_epochs=1,
                                                       filename="checkpoint-{epoch:05d}",
                                                       save_last=True,
                                                       ),
                                       LearningRateMonitor(logging_interval="step"),
                                      ],
                            )
    last_ckpt = os.path.join(args.save_path,"last.ckpt") 
    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)

    trainer.test(model, datamodule=data,
                 ckpt_path=last_ckpt)
    score = trainer.logged_metrics["test_"+model.metric.mode]
    print("test score {}".format(score))
    if (trainer.num_devices > 1):
        print("Note this score is not correct for unevenly distributed data due to DDP")
        print("To get the correct score, please run the script again with --nproc 1")