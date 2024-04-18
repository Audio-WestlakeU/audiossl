from pytorch_lightning import Trainer
from model import FrameATSTLightningModule
from data import FrameATSTDataModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from argparse import ArgumentParser
import os
import torch
import gc



def main(args):
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    args.spec_h = args.n_mels
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")
    #logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    model = FrameATSTLightningModule(**dict_args)
    data = FrameATSTDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            accelerator="gpu",
                            devices=args.nproc,
                            #precision=16,
                            max_steps=args.max_steps,
                            logger=[logger_tb],#,logger_wb],
                            callbacks=[ModelCheckpoint(dirpath=args.save_path,
                                                       every_n_epochs=20,
                                                       filename="checkpoint-{epoch:05d}",
                                                       save_last=True,
                                                       ),
                                       LearningRateMonitor(logging_interval="step"),
                                      ],
                            )
    last_ckpt = os.path.join(args.save_path,"last.ckpt")

    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)

if __name__ == "__main__":
    parser = ArgumentParser("FrameATST")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--save_path",type=str)
    parser.add_argument('--nproc', type=int,  default=2)
    parser.add_argument('--patch_h', type=int,  default=64)
    parser.add_argument('--patch_w', type=int,  default=4)
    parser = FrameATSTLightningModule.add_model_specific_args(parser)
    parser = FrameATSTDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)
