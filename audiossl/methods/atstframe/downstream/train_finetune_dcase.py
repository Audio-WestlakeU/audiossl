
from gc import unfreeze
import os
from argparse import ArgumentParser

import torch
from audiossl import datasets
from audiossl.lightning.datamodules import (DownstreamDataModule,
                                            get_inmemory_datamodule)
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.pyramid.model import FrameATSTLightningModule
from audiossl.methods.pyramid.downstream import utils
from audiossl.datasets.dcase_utils import collate_fn
from audiossl.methods.pyramid.downstream.utils_dcase.model_dcase import (
    FineTuningPLModule, PretrainedEncoderPLModule)
from audiossl.methods.pyramid.downstream.transform import \
    FreezingTransform, FinetuneTargetTransform, FinetuneTrainTransform, FinetuneEvalTransform
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from copy import deepcopy



def get_pretraied_encoder(args):
    # get pretrained encoder
    dict_args = vars(args)
    
    s = torch.load(args.pretrained_ckpt_path)

    pretrained_model = FrameATSTLightningModule.load_from_checkpoint(
        args.pretrained_ckpt_path)

    pretrained_encoder = pretrained_model.model.teacher.encoder
    pretrained_encoder.hyper_param = s['hyper_parameters']
    # [MARK]: Support for 12-block pyramid only
    unfreeze_layers = args.unfreeze_last_n_blocks
    for p in pretrained_encoder.parameters():
        p.requires_grad = False
    # Finetune certain blocks
    unfreeze_ids = list(reversed(range(12)))
    if not unfreeze_ids:
        print("Freeze mode")
        return pretrained_encoder
    else:
        print("Finetune mode")
        unfreeze_ids = unfreeze_ids[: unfreeze_layers]
        print("Unfreezeing last block ids:", unfreeze_ids)
        for i in unfreeze_ids:
            layer = getattr(pretrained_encoder.blocks, str(i))
            for child in layer.parameters():
                child.requires_grad = True
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
    train_transform = FinetuneTrainTransform(max_len=10)
    eval_transform = FinetuneEvalTransform(max_len=10)
    if args.mixup_training:
        target_transform = FinetuneTargetTransform(num_classes=datasets.get_dataset(args.dataset_name).num_labels)
    else:
        target_transform = None

    data = DownstreamDataModule(**dict_args,
                                batch_size=args.batch_size_per_gpu,
                                fold=fold,
                                collate_fn=collate_fn,
                                transforms=[train_transform,eval_transform,eval_transform],
                                target_transforms=[target_transform,None,None])

    """train a linear classifier on extracted embedding"""
    # train
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(save_path, name="tb_logs")
    #logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    num_labels = data.num_labels
    multi_label = data.multi_label
    ckpt_cb = ModelCheckpoint(dirpath=save_path,
                              every_n_epochs=1,
                              filename="checkpoint-{epoch:05d}",
                              save_last=True,
                              monitor="val/object_metric",
                              mode="max",
                              save_top_k=3,
                              )
    model = FineTuningPLModule(
        encoder=pretrained_module,
        num_labels=num_labels,
        multi_label=multi_label,
        niter_per_epoch=len(data.train_dataloader())//args.nproc+1,
        metric_save_dir=(args.save_path),
        **dict_args)

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
    # score = trainer.logged_metrics["test_"+model.metric.mode]
    # print("test score {}".format(score))
    return


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
    parser.add_argument('--nproc', type=int,  default=1)
    parser.add_argument("--dcase_conf", type=str, default="/data/home/lixian/audiossl/audiossl/methods/pyramid/downstream/conf/dcase_dataset.yaml")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=1, help="Unfreeze last n blocks in finetuning.")
    parser = FineTuningPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_folds = dataset_info.num_folds

    """load pretrained encoder"""
    print("Getting pretrain encoder...")
    pretrained_encoder = get_pretraied_encoder(args)
    pretrained_module = PretrainedEncoderPLModule(
        pretrained_encoder,
        args.n_last_blocks
        )
    # pretrained_module.freeze()

    """train"""
    if num_folds > 1:
        run_n_folds(args, pretrained_module, num_folds)
    else:
        run(args, pretrained_module)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()

