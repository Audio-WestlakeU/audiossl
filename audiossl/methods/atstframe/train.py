import json

import yaml
from pytorch_lightning import Trainer
from model import FrameATSTLightningModule
from data import FrameATSTDataModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from argparse import ArgumentParser
import os
import torch
torch.set_float32_matmul_precision('high')

def main(dict_args):
    logger_tb = TensorBoardLogger(dict_args["save_path"],name="tb_logs")
    logger_wb = WandbLogger(save_dir=dict_args["save_path"],name="wb_logs")
    model = FrameATSTLightningModule(**dict_args)
    data = FrameATSTDataModule(**dict_args)
    trainer:Trainer = Trainer(
                            strategy="ddp",
                            sync_batchnorm=True,
                            accelerator="gpu",
                            devices=dict_args["nproc"],
                            #precision=16,
                            max_steps=dict_args["max_steps"],
                            logger=[logger_tb,logger_wb],
                            callbacks=[ModelCheckpoint(dirpath=dict_args["save_path"],
                                                       every_n_epochs=20,
                                                       save_top_k=-1,
                                                       filename="checkpoint-{epoch:3d}",
                                                       save_last=True
                                                       ),
                                       LearningRateMonitor(logging_interval="step"),
                                      ],
                            )
    last_ckpt = os.path.join(dict_args["save_path"],"last.ckpt")

    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)
def parseConfig(configFile):

    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    from pathlib import Path
    parser = ArgumentParser("FrameATST")
    parser.add_argument("--save_path",type=str)
    parser.add_argument('--nproc', type=int,  default=2)
    parser.add_argument('--patch_h', type=int,  default=64)
    parser.add_argument('--patch_w', type=int,  default=4)
    parser = FrameATSTLightningModule.add_model_specific_args(parser)
    parser = FrameATSTDataModule.add_data_specific_args(parser)

    dict_args = vars(parser.parse_args())

    # 用my_train_small.yaml override train_small.sh中的参数
    dict_args.update(parseConfig(configFile=Path(__file__).parent / "music_train_small.yaml"))
    dict_args["spec_h"] = dict_args["n_mels"]

    # 需要根据epoch和batch_size来计算跟step相关的参数
    batch_size = dict_args["nproc"] * dict_args["batch_size_per_gpu"]
    steps_per_epoch = dict_args['ds_size'] / batch_size  # 1912024/864 = 2212
    dict_args['steps_per_epoch'] = steps_per_epoch
    # dict_args["learning_rate"] = dict_args["learning_rate"] * batch_size/256 按这个比例缩小lr，还是overfit，再调小lr
    # 如果yaml文件已经给出steps的参数，就使用给出的参数，不使用epoch
    if 'max_steps' not in dict_args:
        dict_args['max_steps'] = int(steps_per_epoch * dict_args['max_epochs']) # 2212*200=221200
    if 'warmup_steps' not in dict_args:
        dict_args['warmup_steps'] = int(steps_per_epoch * dict_args['warmup_epochs'])
    # 保存这些args到训练的文件夹下。
    with open(os.path.join(dict_args["save_path"], 'args.json'), 'w') as fp:
        json.dump(dict_args, fp)


    # train
    main(dict_args)