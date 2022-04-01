from lib2to3.pgen2.token import AT
from tkinter import W
from pytorch_lightning import Trainer
from model import ATSTLightningModule
from data import ATSTDataModule
from callbacks import CheckpointEveryNEpochs,WeightDecayCosineScheduler
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from argparse import ArgumentParser
import os



def main(args):
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")
    logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    model = ATSTLightningModule(**dict_args)                            
    data = ATSTDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            gpus=args.nproc,
                            max_epochs=301,
                            logger=[logger_tb,logger_wb],
                            callbacks=[ModelCheckpoint(dirpath=args.save_path,
                                                       every_n_epochs=20,
                                                       filename="checkpoint-{epoch:05d}",
                                                       save_last=True,
                                                       ),
                                       LearningRateMonitor(logging_interval="step"),
                                       WeightDecayCosineScheduler(0.04,0.4),
                                      ],
                            )
    last_ckpt = os.path.join(args.save_path,"last.ckpt") 
    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--save_path",type=str)
    parser.add_argument('--nproc', type=int,  default=2)
    parser = ATSTLightningModule.add_model_specific_args(parser)
    parser = ATSTDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)