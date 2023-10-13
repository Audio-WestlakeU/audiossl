
import os
from argparse import ArgumentParser

import torch
from audiossl import datasets
from audiossl.lightning.datamodules import (DownstreamDataModule,
                                            get_inmemory_datamodule)
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.atst.model import ATSTLightningModule
from audiossl.methods.atst.downstream import utils
from audiossl.methods.atst.downstream.data import collate_fn
from audiossl.methods.atst.downstream.model import (
    LinearClassifierPLModule, PretrainedEncoderPLModule)
from audiossl.methods.atst.downstream.transform import \
    FreezingTransform
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from audiossl.models.atst import audio_transformer
from pytorch_lightning import LightningModule


class PretrainedATSTEncoderPLModule(LightningModule):
    def __init__(self,
                 ckpt_path,
                 chunk_len: float,
                 n_blocks: int):
        super().__init__()
        self.encoder = get_pretrained_encoder({"pretrained_ckpt_path":ckpt_path})
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.n_blocks = n_blocks
        self.embed_dim = self.encoder.embed_dim*2*n_blocks

    def forward(self, batch):
        (x, length), y = batch
        x = self.encoder.get_intermediate_layers_chunks(x,
                                                        length,
                                                        self.n_blocks,
                                                        self.chunk_len)
        return x, y


def get_pretrained_encoder(args):
    # get pretrained encoder
    #dict_args = vars(args)

    s = torch.load(args["pretrained_ckpt_path"])

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = ATSTLightningModule.load_from_checkpoint(
            args["pretrained_ckpt_path"])
        pretrained_encoder = pretrained_model.model.teacher.encoder
    else:
        from audiossl.methods.atst.downstream.utils import \
            load_pretrained_weights
        from audiossl.models.atst.audio_transformer import AST_base, AST_small
        pretrained_encoder = AST_small()
        load_pretrained_weights(
            pretrained_encoder, pretrained_weights=args["pretrained_ckpt_path"], checkpoint_key="teacher")
    return pretrained_encoder


def extract_embedding(pretrained_module, data, nproc):
    extracter=EmbeddingExtractor(pretrained_module,nproc=nproc)
    result = extracter.extract(data.train_dataloader())
    result = [r for r in zip(*result)]
    x_train, y_train = result
    x_train = torch.cat(x_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    result = extracter.extract(data.val_dataloader())
    result = [r for r in zip(*result)]
    x_val, y_val = result
    x_val = torch.cat(x_val, dim=0)
    y_val = torch.cat(y_val, dim=0)

    result = extracter.extract(data.test_dataloader())
    result = [r for r in zip(*result)]
    x_test, y_test = result
    x_test = torch.cat(x_test, dim=0)
    y_test = torch.cat(y_test, dim=0)
    return x_train, y_train, x_val, y_val, x_test, y_test


def run(args, pretrained_module, fold=None):
    dict_args = vars(args)

    """extract embedding"""
    transform = FreezingTransform()
    data = DownstreamDataModule(**dict_args,
                                fold=fold,
                                collate_fn=collate_fn,
                                transforms=[transform]*3,
                                limit_batch_size=min(512,args.batch_size_per_gpu))
    x_train, y_train, x_val, y_val, x_test, y_test = extract_embedding(pretrained_module,
                                                                       data,
                                                                       args.nproc)

    """train a linear classifier on extracted embedding"""
    if fold is None or fold == 1:
        args.learning_rate = args.learning_rate*args.nproc*args.batch_size_per_gpu/256
    if fold is not None:
        save_path = os.path.join(args.save_path, "fold{}".format(fold))
    else:
        save_path = args.save_path
    # train
    dict_args = vars(args)
    logger_tb = TensorBoardLogger(save_path, name="tb_logs")
    #logger_wb = WandbLogger(save_dir=args.save_path,name="wb_logs")
    embed_dim = x_train.shape[1]
    num_labels = data.num_labels
    multi_label = data.multi_label

    inmemory_datamodule = get_inmemory_datamodule(x_train,
                                                  y_train,
                                                  x_val,
                                                  y_val,
                                                  x_test,
                                                  y_test,
                                                  args.batch_size_per_gpu)

    model = LinearClassifierPLModule(
        embed_dim=embed_dim,
        num_labels=num_labels,
        multi_label=multi_label,
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
        max_epochs=args.max_epochs,
        logger=logger_tb,  # ,logger_wb],
        callbacks=[
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    last_ckpt = os.path.join(save_path, "last.ckpt")
    trainer.fit(model, datamodule=inmemory_datamodule,
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
    trainer.test(model, datamodule=inmemory_datamodule,
                 ckpt_path=ckpt_cb.best_model_path)
    score = trainer.logged_metrics["test_"+model.metric.mode]
    print("test score {}".format(score))
    return score


def run_n_folds(args, pretrained_module, num_folds):
    test_metrics = []
    for fold in range(num_folds):
        test_metrics.append(run(args, pretrained_module, fold+1))
    test_metrics = torch.stack(test_metrics)
    avg = torch.mean(test_metrics)
    print("{} folds's test scores:{}".format(num_folds, test_metrics))
    print("average test score:{}".format(avg))


def main():
    parser = ArgumentParser("LinearClassifier")
    #parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--n_last_blocks", type=int, default=12)
    parser.add_argument("--pretrained_ckpt_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument('--nproc', type=int,  default=1)
    parser = LinearClassifierPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    args = parser.parse_args()

    dataset_info = datasets.get_dataset(args.dataset_name)
    num_folds = dataset_info.num_folds

    """load pretrained encoder"""
    pretrained_encoder = get_pretrained_encoder(args)
    pretrained_module = PretrainedEncoderPLModule(pretrained_encoder,
                                                        6.,
                                                        args.n_last_blocks)
    pretrained_module.freeze()

    """train"""
    if num_folds > 1:
        run_n_folds(args, pretrained_module, num_folds)
    else:
        run(args, pretrained_module)

import hydra
from omegaconf import DictConfig,OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_info
@hydra.main(config_path="./conf",config_name="config")
def main_hydra(cfg:DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg))
    print(cfg.data)
    print(cfg.model)
    print(cfg.model.pretrained)
    pretrained_module = hydra.utils.instantiate(cfg.model.pretrained)
    print("")

if __name__ == "__main__":
    main_hydra()
