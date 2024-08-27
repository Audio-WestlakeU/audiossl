import torch
from audiossl.methods.atstframe.model import FrameATSTLightningModule
from audiossl.methods.atst.downstream.train_freeze import get_pretraied_encoder
from argparse import ArgumentParser
from audiossl.methods.atstframe.module_distill import DistillATSTDataModule,DistillLightningModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
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
    #parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--pretrained_ckpt_path_clip", type=str)
    parser.add_argument("--pretrained_ckpt_path_frame", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = DistillLightningModule.add_model_specific_args(parser)
    parser = DistillATSTDataModule.add_data_specific_args(parser)

    args = parser.parse_args()
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")

    #load model
    #cls_encoder = get_pretraied_encoder(args)
    cls_s = torch.load(args.pretrained_ckpt_path_clip,map_location="cpu")
    frame_encoder = get_pretraied_encoder_frame(args)
    model = DistillLightningModule(**dict_args)                            

    state_dict_cls = cls_s["state_dict"]
    state_dict_cls = {k.replace("encoder.encoder","encoder"):v for k,v in state_dict_cls.items()}
    r1=model.model.teacher.load_state_dict(state_dict_cls,strict=False)
    r2=model.model.student.load_state_dict(frame_encoder.state_dict(),strict=False)
    for n,p in model.model.teacher.named_parameters():
        p.requires_grad = False

    data = DistillATSTDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp_find_unused_parameters_true",
                            sync_batchnorm=True,
                            accelerator="gpu",
                            devices=args.nproc,
                            max_steps=args.max_steps,
                            gradient_clip_val=3.0,
                            logger=[logger_tb],
                            use_distributed_sampler=False,
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