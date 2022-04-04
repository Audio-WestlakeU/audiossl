
from pytorch_lightning import Trainer
from model import FreezingTransferLightningModule,PretrainedEncoderLightningModel
from data import FreezingTransferDataModule
from audiossl.methods.atst.model import ATSTLightningModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from argparse import ArgumentParser
import torch
import os


def extract_embedding(args):
    #get pretrained encoder
    dict_args = vars(args)
    pretrained_model = ATSTLightningModule.load_from_checkpoint(args.pretrained_ckpt_path)
    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_module = PretrainedEncoderLightningModel(pretrained_encoder,6.,12)

    data = FreezingTransferDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            gpus=args.nproc,
                            max_epochs=1,
                            )
    result=trainer.predict(pretrained_module,data.train_dataloader())
    result = [r for r in zip(*result)]
    x_train,y_train = result
    x_train = torch.cat(x_train,dim=0)
    y_train = torch.cat(y_train,dim=0)
    
    result=trainer.predict(pretrained_module,data.val_dataloader())
    result = [r for r in zip(*result)]
    x_val,y_val = result
    x_val = torch.cat(x_val,dim=0)
    y_val = torch.cat(y_val,dim=0)

    result=trainer.predict(pretrained_module,data.test_dataloader())
    result = [r for r in zip(*result)]
    x_test,y_test = result
    x_test = torch.cat(x_test,dim=0)
    y_test = torch.cat(y_test,dim=0)
    return x_train,y_train,x_val,y_val,x_test,y_test


def main(args):
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")
    logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    r=extract_embedding(args)
    return r

    data = FreezingTransferDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            gpus=args.nproc,
                            max_steps=args.max_steps,
                            logger=[logger_tb,logger_wb],
                            callbacks=[ModelCheckpoint(dirpath=args.save_path,
                                                       every_n_epochs=20,
                                                       filename="checkpoint-{epoch:05d}",
                                                       save_last=True,
                                                       ),
                                       LearningRateMonitor(logging_interval="step"),
                                      ],
                            )
    model = FreezingTransferLightningModule(**dict_args)                            
    last_ckpt = os.path.join(args.save_path,"last.ckpt") 
    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)

if __name__ == "__main__":
    parser = ArgumentParser("FreezingTransfer")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--pretrained_ckpt_path",type=str)
    parser.add_argument("--save_path",type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = FreezingTransferLightningModule.add_model_specific_args(parser)
    parser = FreezingTransferDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    # train
    r=main(args)
