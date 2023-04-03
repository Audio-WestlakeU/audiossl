
import os
from argparse import ArgumentParser

import torch
from audiossl import datasets
from audiossl.lightning.datamodules import (DownstreamDataModule,
                                            get_inmemory_datamodule)
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.promptpool.model import FrameATSTLightningModule
from audiossl.methods.promptpool.model_cls import ClsPromptLightningModule
from audiossl.methods.promptpool.downstream import utils
from audiossl.methods.promptpool.downstream.data import collate_fn

from audiossl.methods.promptpool.downstream.model import (
    FineTuningPLModule, PretrainedEncoderPLModule, PretrainedCLsPromptEncoderPLModule, PretrainedPromptPoolEncoderPLModule)
from audiossl.methods.promptpool.downstream.transform import \
    FreezingTransform, FinetuneTargetTransform, FinetuneTrainTransform, FinetuneEvalTransform
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from audiossl.methods.promptpool.model_promptpool import PromptPoolLightningModule
from copy import deepcopy
from audiossl.methods.promptpool.downstream.query import extract_queries

def get_pretraied_encoder(args):
    # get pretrained encoder
    dict_args = vars(args)

    s = torch.load(args.pretrained_ckpt_path)

    if 'pool_size' in s['hyper_parameters']:
        pretrained_model = PromptPoolLightningModule.load_from_checkpoint(
            args.pretrained_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
        return pretrained_encoder
    elif 'nprompt' in s['hyper_parameters']:
        pretrained_model = ClsPromptLightningModule.load_from_checkpoint(
            args.pretrained_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
        return pretrained_encoder


    else:
        pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
            args.pretrained_ckpt_path)

        pretrained_encoder = pretrained_model.model.teacher.encoder
    return pretrained_encoder



def run(args, pretrained_module, fold=None):
    dict_args = vars(args)
    if fold is None or fold == 1:
        args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    if fold is not None:
        save_path = os.path.join(args.save_path, "fold{}".format(fold))
    else:
        save_path = args.save_path

    """extract embedding"""
    train_transform = FinetuneTrainTransform()
    eval_trainsform = FinetuneEvalTransform()
    if args.mixup_training:
        target_transform = FinetuneTargetTransform(num_classes=datasets.get_dataset(args.dataset_name).num_labels)
    else:
        target_transform = None
    data = DownstreamDataModule(**dict_args,
                                batch_size=args.batch_size_per_gpu,
                                fold=fold,
                                collate_fn=collate_fn,
                                transforms=[train_transform,eval_trainsform,eval_trainsform],
                                target_transforms=[target_transform,None,None],
                                return_key=True)
    # query hack
    def query_():
        queries,queries_train = extract_queries("/data/home/lixian/audiossl/ckpts/base.ckpt",data,1)
        torch.save([queries,queries_train],os.path.join(args.save_path,"query.ckpt"))
        return queries,queries_train
    
    if os.path.exists(os.path.join(args.save_path,"query.ckpt")):
        queries,queries_train = torch.load(os.path.join(args.save_path,"query.ckpt"))
    else:
        queries,queries_train = query_()

    pretrained_module.queries = queries
    pretrained_module.queries_train = queries_train
    """train a linear classifier on extracted embedding"""
    # train
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(save_path, name="tb_logs")
    #logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    num_labels = data.num_labels

    multi_label = data.multi_label

    model = FineTuningPLModule(
        encoder=pretrained_module,
        num_labels=num_labels,
        multi_label=multi_label,
        niter_per_epoch=len(data.train_dataloader())//args.nproc+1,
        **dict_args)
    ckpt_cb = ModelCheckpoint(dirpath=save_path,
                              every_n_epochs=1,
                              filename="checkpoint-{epoch:05d}",
                              save_last=True,
                              monitor="val_" + model.metric.mode,
                              mode="max",
                              save_top_k=10,
                              )
    trainer: Trainer = Trainer(
        strategy="ddp",
        sync_batchnorm=True,
        gpus=args.nproc,
        gradient_clip_val=3.0,
        max_epochs=args.max_epochs,
        logger=logger_tb,  # ,logger_wb],
        callbacks=[
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    last_ckpt = os.path.join(save_path, "last.ckpt")
    trainer.fit(model, datamodule=data,
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
    trainer.test(model, datamodule=data,
                 ckpt_path=ckpt_cb.best_model_path)
    score = trainer.logged_metrics["test_"+model.metric.mode]
    print("test score {}".format(score))
    return score


def run_n_folds(args, pretrained_module, num_folds):
    test_metrics = []
    for fold in range(num_folds):
        test_metrics.append(run(args, deepcopy(pretrained_module), fold+1))
    test_metrics = torch.stack(test_metrics)
    avg = torch.mean(test_metrics)
    print("{} folds's test scores:{}".format(num_folds, test_metrics))
    print("average test score:{}".format(avg))


def main():
    parser = ArgumentParser("FineTuning")
    #parser = Trainer.add_argparse_args(parser)
    from audiossl.utils.common import bool_flag

    parser.add_argument("--n_last_blocks", type=int, default=12)
    parser.add_argument("--finetune_n_last_blocks", type=int, default=1)
    parser.add_argument("--pretrained_ckpt_path", type=str,required=True)
    parser.add_argument("--samplewise_query", type=bool_flag, default=False)
    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--save_path", type=str,required=True)
    parser.add_argument("--mixup_training", type=bool_flag,default=False)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = FineTuningPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_folds = dataset_info.num_folds

    """load pretrained encoder"""
    pretrained_encoder = get_pretraied_encoder(args)
    if hasattr(pretrained_encoder,"pool_size"):
        pretrained_module = PretrainedPromptPoolEncoderPLModule(pretrained_encoder,
                                                        None,
                                                        6.,
                                                        args.n_last_blocks,
                                                        samplewise_query=args.samplewise_query,
                                                        scene=True if args.head=="linear" else False)
    elif hasattr(pretrained_encoder,"nprompt"):
        pretrained_module = PretrainedCLsPromptEncoderPLModule(pretrained_encoder,
                                                        6.,
                                                        args.n_last_blocks)
    else:
        pretrained_module = PretrainedEncoderPLModule(pretrained_encoder,
                                                        6.,
                                                        args.n_last_blocks)
    #pretrained_module.unfreeze()

    for i in range(args.finetune_n_last_blocks):
        block = pretrained_module.encoder.framemodel.blocks[-1-i]
        for param in block.parameters():
            param.requires_grad=True

    pretrained_module.train()
    pretrained_module.encoder.prompt_hgram.requires_grad=False
    pretrained_module.encoder.prompt_keys.requires_grad=False

    """train"""
    if num_folds > 1:
        run_n_folds(args, pretrained_module, num_folds)
    else:
        run(args, pretrained_module)


if __name__ == "__main__":
    main()

