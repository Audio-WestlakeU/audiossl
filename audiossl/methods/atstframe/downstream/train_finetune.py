
import os
from argparse import ArgumentParser

import torch
from audiossl import datasets
from audiossl.lightning.datamodules import (DownstreamDataModule,
                                            get_inmemory_datamodule)
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.atstframe.model import FrameATSTLightningModule
from audiossl.methods.atstframe.downstream import utils
from audiossl.methods.atstframe.downstream.data import collate_fn
from audiossl.methods.atstframe.downstream.model import (
    FineTuningPLModule, PretrainedEncoderPLModule)
from audiossl.methods.atstframe.downstream.transform import \
    FreezingTransform, FinetuneTargetTransform, FinetuneTrainTransform, FinetuneEvalTransform,FinetuneTargetTransformAudioset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from copy import deepcopy
import numpy as np

def get_pretraied_encoder(args):
    # get pretrained encoder
    dict_args = vars(args)

    s = torch.load(args.pretrained_ckpt_path,map_location="cpu")
    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        args.pretrained_ckpt_path)

    if args.use_encoder == "teacher":
        pretrained_encoder = pretrained_model.model.teacher.encoder
    else:
        pretrained_encoder = pretrained_model.model.student.encoder

    pretrained_encoder.hyper_param = s['hyper_parameters']
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
    train_transform = FinetuneTrainTransform(is_roll_mag=args.roll_mag)
    eval_trainsform = FinetuneEvalTransform()
    #if "audioset" in args.dataset_name:
    #    train_transform = FinetuneTrainTransform(max_len=10,pad=True)
    #    eval_trainsform = FinetuneEvalTransform(max_len=10, pad=True)
    if args.mixup_training:
        target_transform = FinetuneTargetTransform(num_classes=datasets.get_dataset(args.dataset_name).num_labels,
                                                   alpha=args.alpha)
    else:
        target_transform = None
    if args.dataset_name == "audioset":
        s = torch.load(os.path.join(args.data_path,"weights_labels.pt"),map_location="cpu")
        weights = s["weights_labels"]
        keys = s["keys"]


        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, len(weights))
        data_ = DownstreamDataModule(**dict_args,
                                    batch_size=args.batch_size_per_gpu,
                                    fold=fold,
                                    collate_fn=collate_fn,
                                    transforms=[train_transform,eval_trainsform,eval_trainsform],
                                    target_transforms=[None,None,None],
                                    sampler=None)
        target_transform = FinetuneTargetTransformAudioset(dataset=data_.dataset_train,
                                                           is_mask_aug= args.mask_aug,
                                                           is_rrc = args.rrc,
                                                           num_classes=datasets.get_dataset(args.dataset_name).num_labels,
                                                           mixup_ratio=args.mixup_ratio,
                                                           alpha=args.alpha,)

        data = DownstreamDataModule(**dict_args,
                                    batch_size=args.batch_size_per_gpu,
                                    fold=fold,
                                    collate_fn=collate_fn,
                                    transforms=[train_transform,eval_trainsform,eval_trainsform],
                                    target_transforms=[target_transform,None,None],
                                    sampler=sampler)
        #data.dataset_train.keys = keys
    else:
        data = DownstreamDataModule(**dict_args,
                                batch_size=args.batch_size_per_gpu,
                                fold=fold,
                                collate_fn=collate_fn,
                                transforms=[train_transform,eval_trainsform,eval_trainsform],
                                target_transforms=[target_transform,None,None])

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
        niter_per_epoch=len(data.dataset_train)//args.batch_size_per_gpu//args.nproc+1,
        layer_wise_lr = args.layerwise_lr,
        #optimizer= "adamw" if args.dataset_name == "audioset" else "SGD",
        **dict_args)
    ckpt_cb = ModelCheckpoint(dirpath=save_path,
                              every_n_epochs=1,
                              filename="checkpoint-{epoch:05d}",
                              save_last=True,
                              monitor="val_" + model.metric.mode if "audioset" in args.dataset_name else None,
                              mode="max",
                              save_top_k=10 if "audioset" in args.dataset_name else 1,
                              )
    trainer: Trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
        accelerator="gpu",
        devices=args.nproc,
        gradient_clip_val=3.0,
        max_epochs=args.max_epochs,
        logger=logger_tb,  # ,logger_wb],
        use_distributed_sampler=False if args.dataset_name == "audioset" else True,
        check_val_every_n_epoch=1,
        #val_check_interval=1 if args.dataset_name == "audioset" else 1.0,
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
    if (trainer.num_devices > 1):
        print("Note this score is not correct for unevenly distributed data due to DDP")
        print("To get the correct score, please run the script again with --nproc 1")
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
    parser.add_argument("--pretrained_ckpt_path", type=str,required=True)
    parser.add_argument("--save_path", type=str,required=True)
    parser.add_argument("--mixup_training", type=bool_flag,default=False)
    parser.add_argument("--layerwise_lr", type=float,default=0.75)
    parser.add_argument("--mask_aug", type=bool_flag,default=False)
    parser.add_argument("--freeze_embed", type=bool_flag,default=False)
    parser.add_argument("--rrc", type=bool_flag,default=False)
    parser.add_argument('--nproc', type=int,  default=1)
    parser.add_argument('--mixup_ratio', type=float,  default=1.0)
    parser.add_argument('--alpha', type=float,  default=10)
    parser.add_argument("--roll_mag", type=bool_flag,default=False)
    parser.add_argument('--use_encoder', type=str,  default="teacher")
    parser = FineTuningPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_folds = dataset_info.num_folds

    """load pretrained encoder"""
    pretrained_encoder = get_pretraied_encoder(args)
    pretrained_module = PretrainedEncoderPLModule(pretrained_encoder,
                                                    6.,
                                                    args.n_last_blocks)
    pretrained_module.unfreeze()

    """train"""
    if num_folds > 1:
        run_n_folds(args, pretrained_module, num_folds)
    else:
        run(args, pretrained_module)


if __name__ == "__main__":
    main()

