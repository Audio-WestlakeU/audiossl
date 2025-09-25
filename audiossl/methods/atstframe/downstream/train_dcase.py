
from gc import unfreeze
import os
from argparse import ArgumentParser
from audiossl import datasets
from audiossl.lightning.frame_datamodules import DownstreamDataModule
from audiossl.methods.atstframe.downstream.comparison_models.ssast_module import SSASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.byola_module import BYOLAPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mae_ast_module import MAEASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.clip_atst_module import ATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.frame_atst_module import FrameATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.distill_atst_module import DistillATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.patch_ssast_module import PatchSSASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mae_ast_patch_module import PatchMAEASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.beats_module import BeatsPredModule
from audiossl.methods.atstframe.downstream.comparison_models.audioMAE_module import AudioMAEPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mmd_module import MMDPredModule
from audiossl.datasets.dcase_utils import collate_fn
from audiossl.methods.atstframe.downstream.utils_dcase.model_distill import DistillPLModule
from audiossl.methods.atstframe.downstream.utils_dcase.model_dcase import FineTuningPLModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def run(args, pretrained_module):
    dict_args = vars(args)
    test_ckpt = args.test_from_checkpoint
    save_path = args.save_path

    """extract embedding"""
    train_transform = pretrained_module.transform
    eval_transform = pretrained_module.transform
    target_transform = None

    data = DownstreamDataModule(**dict_args,
                                batch_size=args.batch_size_per_gpu,
                                fold=False,
                                collate_fn=collate_fn,
                                transforms=[train_transform, eval_transform, eval_transform],
                                target_transforms=[target_transform,None,None],
                                ignores=["transforms"])

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
    if args.arch == "distillatst":
        model = DistillPLModule(
            encoder=pretrained_module,
            num_labels=num_labels,
            multi_label=multi_label,
            niter_per_epoch=len(data.train_dataloader()),
            metric_save_dir=(args.save_path),
            learning_rate=args.learning_rate,
            dcase_conf=args.dcase_conf,
            max_epochs=args.max_epochs,
            warmup_epochs=args.warmup_epochs,
            freeze_mode=args.freeze_mode,
            distill_mode=args.distill_mode
            )
    else:
        model = FineTuningPLModule(
            encoder=pretrained_module,
            num_labels=num_labels,
            multi_label=multi_label,
            niter_per_epoch=len(data.train_dataloader()),
            metric_save_dir=(args.save_path),
            learning_rate=args.learning_rate,
            dcase_conf=args.dcase_conf,
            max_epochs=args.max_epochs,
            warmup_epochs=args.warmup_epochs,
            freeze_mode=args.freeze_mode,
            )

    trainer: Trainer = Trainer(
        strategy="auto",
        sync_batchnorm=True,
        devices=args.nproc,
        gradient_clip_val=3.0,
        max_epochs=args.max_epochs,
        logger=logger_tb,  # ,logger_wb],
        callbacks=[
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    last_ckpt = os.path.join(save_path, "last.ckpt")
    if test_ckpt is None:
        trainer.fit(model, datamodule=data,
                    ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
        best_model = ckpt_cb.best_model_path
    else:
        best_model = test_ckpt
    trainer.test(model, datamodule=data,
                 ckpt_path=best_model)
    # score = trainer.logged_metrics["test_"+model.metric.mode]
    # print("test score {}".format(score))
    return

def main():
    parser = ArgumentParser("FineTuning")
    #parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--arch', type=str,  default="ssast")
    parser.add_argument("--pretrained_ckpt_path", type=str, default="./comparison_models/ckpts/SSAST-Base-Frame-400.pth")
    parser.add_argument("--save_path", type=str, default="./logs/dcase_10/")
    parser.add_argument('--nproc', type=str,  default="1,")
    parser.add_argument("--dcase_conf", type=str, default="./utils_dcase/conf/patch_160.yaml")
    parser.add_argument("--test_from_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_mode", action="store_true")
    parser.add_argument("--prefix", type=str, default="/")
    parser = FineTuningPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)
    
    args = parser.parse_args()
    if not args.freeze_mode:
        args.prefix += "_finetune/"
    # Change log name
    args.save_path = args.save_path + args.arch + args.prefix
    
        
    # Registry dataset
    args.dataset_name = "dcase"
    dataset_info = datasets.get_dataset(args.dataset_name)
    # Read config files and overwrite setups
    """load pretrained encoder"""
    print("Getting pretrain encoder...")
    if args.arch == "ssast":
        pretrained_module = SSASTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "byola":
        pretrained_module = BYOLAPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "maeast":
        pretrained_module = MAEASTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "clipatst":
        pretrained_module = ATSTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "frameatst":
        pretrained_module = FrameATSTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "distillatst":
        pretrained_module = DistillATSTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "patchssast":
        pretrained_module = PatchSSASTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "patchmaeast":
        pretrained_module = PatchMAEASTPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "beats":
        pretrained_module = BeatsPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "audioMAE":
        pretrained_module = AudioMAEPredModule(args.pretrained_ckpt_path, args.dataset_name)
    elif args.arch == "mmd":
        pretrained_module = MMDPredModule(args.pretrained_ckpt_path, args.dataset_name)
    print("Freezing/Unfreezing encoder parameters?...", end="")

    if args.freeze_mode:
        print("Freeze mode")
        pretrained_module.freeze()
    else:
        print("Finetune mode")
        pretrained_module.finetune_mode()
    run(args, pretrained_module)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()