from pytorch_lightning import Trainer
from model_promptpool import PromptPoolLightningModule
from data_promptpool import PromptPoolDataModule
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger,WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero
from argparse import ArgumentParser
from query import extract_queries,QueryDataModule,kmeans_cosine
import torch
import os
from fast_pytorch_kmeans import KMeans
from transform import FrameATSTTrainTransform

from transform_cls import ClsPromptTrainTransform

    


def main(args):
    args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(args.save_path,name="tb_logs")
    logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")

    @rank_zero.rank_zero_only
    def query_():
        data = QueryDataModule(**dict_args)
        queries,x = extract_queries(args.query_model,data,1)#args.nproc)
        keys = queries.keys()
        kmeans = KMeans(n_clusters=args.pool_size,max_iter=100,mode="cosine",verbose=1)

        kmeans.fit(x.to("cuda"))

        c = kmeans.centroids.cpu()

        #cl,c = kmeans_cosine(x.to("cuda"),K=args.pool_size)
        #c = c.cpu()
        #cl,c = kmeans_cosine(x.to("cuda"),K=args.pool_size)
        torch.save([queries,c],os.path.join(args.save_path,"query.ckpt"))
        return queries,c
    
    if os.path.exists(os.path.join(args.save_path,"query.ckpt")):
        queries,c = torch.load(os.path.join(args.save_path,"query.ckpt"))
    else:
        queries,c = query_()

    transform = None
    if args.train_mode=="cls":
        transform = ClsPromptTrainTransform()
    else:
        transform = FrameATSTTrainTransform()

    data = PromptPoolDataModule(queries=queries,transform=transform,**dict_args)
    model = PromptPoolLightningModule(prompt_key=c.cpu(),**dict_args)                            
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
    last_ckpt = os.path.join(args.save_path,"last.ckpt") 

    if (not args.stage1_ckpt_path == "None") and (not os.path.exists(last_ckpt)):
        #load stage1 model
        s = torch.load(args.stage1_ckpt_path)
        model.load_state_dict(s['state_dict'],strict=False)
        model.model._init_teacher()
        print("======================load weights from {}",args.stage1_ckpt_path)

    trainer.fit(model,datamodule=data,
                ckpt_path=last_ckpt if  os.path.exists(last_ckpt) else None)

if __name__ == "__main__":
    parser = ArgumentParser("ATST")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--stage1_ckpt_path",type=str,default="None")
    parser.add_argument("--train_mode",type=str,default="cls")
    parser.add_argument("--save_path",type=str)
    parser.add_argument("--query_model",type=str)
    parser.add_argument('--nproc', type=int,  default=2)
    parser = PromptPoolLightningModule.add_model_specific_args(parser)
    parser = PromptPoolDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)