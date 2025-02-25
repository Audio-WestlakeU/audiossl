import os
from argparse import ArgumentParser
from pytorch_lightning.strategies.ddp import DDPStrategy
from audiossl.lightning.frame_datamodules import DownstreamDataModule
import pytorch_lightning as pl
from audiossl.methods.atstframe.downstream.comparison_models.distill_atst_module import DistillATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mert_module import MertPredModule
from audiossl.methods.atstframe.downstream.comparison_models.ssast_module import SSASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.byola_module import BYOLAPredModule
from audiossl.methods.atstframe.downstream.comparison_models.clip_atst_module import ATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mae_ast_module import MAEASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.patch_ssast_module import PatchSSASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mae_ast_patch_module import PatchMAEASTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.frame_atst_module import FrameATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.beats_module import BeatsPredModule
from audiossl.methods.atstframe.downstream.comparison_models.audioMAE_module import AudioMAEPredModule
from audiossl.methods.atstframe.downstream.comparison_models.mmd_module import MMDPredModule
from audiossl.datasets.dcase_utils import collate_fn
from audiossl.methods.atstframe.downstream.utils_as_strong.model_as_strong import FineTuningPLModule
from audiossl.methods.atstframe.downstream.utils_as_strong.model_distill_as_strong import DistillPLModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
import torch
from datetime import datetime

torch.set_float32_matmul_precision('high')
current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H-%M")  # 用于此次wandbproject名称
pl.seed_everything(42)  # 当设置45的时候，发现dataloader读取标签出来，抛弓竟然为0，随机种子能够影响数据读取？！

def run(dict_args, pretrained_module, kth_fold, save_path, test_ckpt=""):
    os.makedirs(save_path, exist_ok=True)

    """extract embedding"""
    data_transform = pretrained_module.transform

    data = DownstreamDataModule(**dict_args,
                                kth_fold=kth_fold,
                                batch_size=dict_args["batch_size_per_gpu"],
                                fold=False,
                                collate_fn=collate_fn,
                                transforms=[data_transform, data_transform, data_transform],
                                target_transforms=[None, None, None],
                                ignores=["transforms"])

    """train a linear classifier on extracted embedding"""
    # train
    num_labels = data.num_labels
    multi_label = data.multi_label
    ckpt_cb = ModelCheckpoint(dirpath=save_path,
                              every_n_epochs=1,
                              filename="checkpoint-{epoch:03d}",
                              # save_last=True,
                              monitor="val/loss",
                              mode="min",
                              save_top_k=1,
                              )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=10,
        verbose=True,
        mode="min",
    )

    # 从训练数据计算loss weights
    loss_weights = get_weights_for_loss(data.train_dataloader(), param=dict_args['loss_weight_param'])

    if dict_args["arch"] == "distill":
        model = DistillPLModule(
            encoder=pretrained_module,
            num_labels=num_labels,
            multi_label=multi_label,
            niter_per_epoch=len(data.train_dataloader()) // dict_args["nproc"],
            metric_save_dir=save_path,
            learning_rate=dict_args["learning_rate"],
            dcase_conf=dict_args["dcase_conf"],
            max_epochs=dict_args["max_epochs"],
            warmup_epochs=dict_args["warmup_epochs"],
            freeze_mode=dict_args["freeze_mode"],
            lr_scale=dict_args["lr_scale"],
            distill_mode=dict_args["pretrained_ckpt_path"]
        )
    else:
        model = FineTuningPLModule(
            encoder=pretrained_module,
            num_labels=num_labels,
            multi_label=multi_label,
            niter_per_epoch=len(data.train_dataloader()) // dict_args["nproc"],
            metric_save_dir=save_path,
            learning_rate=dict_args["learning_rate"],
            dcase_conf=dict_args["dcase_conf"],
            max_epochs=dict_args["max_epochs"],
            warmup_epochs=dict_args["warmup_epochs"],
            freeze_mode=dict_args["freeze_mode"],
            lr_scale=dict_args["lr_scale"],
            loss_weights=loss_weights,
            classifier=dict_args["classifier"],
            focal_gamma=dict_args['focal_gamma']
        )
    strategy = 'auto' if dict_args["nproc"] == 1 else DDPStrategy(find_unused_parameters=False)
    # 初始化 WandB 运行
    if test_ckpt == "":
        wandb.init(
            project=f"audiossl_{time_str}",
            group="5-fold",  # 将所有实验分组
            name=f"k_fold_{kth_fold + 1}"  # 每次实验的名称
        )
    wandb_logger = WandbLogger() if test_ckpt == "" else None
    trainer: Trainer = Trainer(
        strategy=strategy,
        num_sanity_val_steps=3,
        # flush_logs_every_n_steps=10, This param has been deprecated.
        sync_batchnorm=False,
        accelerator="gpu",
        devices=dict_args["nproc"],
        gradient_clip_val=3.0,
        max_epochs=dict_args["max_epochs"],
        logger=wandb_logger,
        callbacks=[
            ckpt_cb,
            early_stop_cb,
            LearningRateMonitor(logging_interval="step")
        ],
        # limit_test_batches=5,
    )
    last_ckpt = os.path.join(save_path, "last.ckpt")
    if test_ckpt == "":
        trainer.fit(model, datamodule=data,
                    ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None)
        best_val_loss = ckpt_cb.best_model_score.item()
        wandb.finish()
        return best_val_loss
    else:
        best_ckpt = test_ckpt
        trainer.test(model, datamodule=data, ckpt_path=best_ckpt)


def parseConfig(configFile):
    import yaml
    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_weights_for_loss(dataloader, param):
    print(f"loss_weight param is {param}")
    if param == 0:
        return None
    # extract labels
    all_labels = []
    for x, labels, _ in dataloader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)  # (64 * no_of_batches, 7, 250) = (704, 8, 250)

    print('labels: ', all_labels.shape)  # (704, 8, 250)
    class_sample_counts = all_labels.sum(0).sum(1)  # 对704和250求和 形状: (8,)
    print('class_sample_counts:', class_sample_counts)

    # label只记录了有技法的类别，需要手动把NA类别也加上
    total_samples = all_labels.size(0) * all_labels.size(2)  # 共有这么多time_steps
    print('total_samples: ', total_samples)
    total_samples_na = total_samples - class_sample_counts.sum()  # 这些是NA的类别数
    print('total_samples of NA: ', total_samples_na)
    class_sample_counts[0] = total_samples_na  # 第一个index为0的就是NA的个数

    class_frequencies = class_sample_counts.float() / class_sample_counts.sum()
    print('class_frequencies:', class_frequencies)
    class_weights = ((class_frequencies.mean() / class_frequencies) *  # 频率均值与每个类别频率的比值
                     ((1 - class_frequencies) / (1 - class_frequencies.mean()))  # 频率补数与均值补数的比值
                     ) ** param  # 平滑处理
    print('class_weights:', class_weights)
    return class_weights


def main():
    from pathlib import Path
    import json
    parser = ArgumentParser("FineTuning")
    # parser.add_argument('--arch', type=str,  default="ssast")
    # parser.add_argument("--pretrained_ckpt_path", type=str, default=".comparison_models/ckpts/SSAST-Base-Frame-400.pth")
    # parser.add_argument("--save_path", type=str, default="./logs/as_strong_407/")
    # parser.add_argument('--nproc', type=str,  default="1,")
    # parser.add_argument("--dcase_conf", type=str, default="./conf/patch_40.yaml")
    # parser.add_argument("--test_from_checkpoint", type=str, default=None)
    # parser.add_argument("--freeze_mode", action="store_true")
    # parser.add_argument("--prefix", type=str, default="/")
    # parser.add_argument("--lr_scale", type=float, default=1.0)
    parser = FineTuningPLModule.add_model_specific_args(parser)
    parser = DownstreamDataModule.add_data_specific_args(parser)

    dict_args = vars(parser.parse_args())

    # 用my_train_small.yaml override train_small.sh中的参数
    dict_args.update(parseConfig(
        configFile="/20A021/projects/audiossl/audiossl/methods/atstframe/shell/downstream/finetune_as_strong/finetune_frame_atst.yaml"))
    print("Target task:", dict_args["dataset_name"])
    pretrained_ckpt_path = dict_args["pretrained_ckpt_path"]

    # 训练模式
    if "test_from_checkpoints" not in dict_args:
        config = parseConfig(dict_args["dcase_conf"])
        dict_args["dcase_conf_params"] = config
        with open(os.path.join(dict_args["save_path"], 'args.json'), 'w') as fp:
            json.dump(dict_args, fp)
        run_k_fold(dict_args, pretrained_ckpt_path)
    else:
        predict_k_fold(dict_args, pretrained_ckpt_path)


def run_k_fold(dict_args, pretrained_ckpt_path):
    print(f"Start run {dict_args['k_fold']}_fold cross_validation")
    train_data_list = dict_args['dcase_conf_params']['data']['strong_train_k_fold']
    val_data_list = dict_args['dcase_conf_params']['data']['strong_val_k_fold']
    assert (len(train_data_list) == dict_args['k_fold'])
    assert (len(val_data_list) == dict_args['k_fold'])
    best_val_loss_list = []
    arch = dict_args['arch']
    for k in range(dict_args['k_fold']):
        print(
            f'-----------------------------------------------------Run {k + 1}_fold---------------------------------------------------')
        print("Getting pretrain encoder...")
        if arch == "frameatst":
            pretrained_module = FrameATSTPredModule(pretrained_ckpt_path,
                                                    finetune_layer=dict_args["finetune_layer"],
                                                    use_last=dict_args["use_last"],
                                                    drop_rate=0.0, attn_drop_rate=0.0)
        elif arch == "mert":
            pretrained_module = MertPredModule(pretrained_ckpt_path)
        else:
            raise NotImplementedError(f"{arch} not supported!")
        print("Freezing/Unfreezing encoder parameters?...", end="")
        if dict_args["freeze_mode"]:
            print("Freeze mode")
            pretrained_module.freeze()
        else:
            print("Finetune mode")
            pretrained_module.finetune_mode()
        best_val_loss = run(dict_args, pretrained_module, k,
                            save_path=os.path.join(dict_args["save_path"], f"fold_{k + 1}"))
        best_val_loss_list.append(best_val_loss)
    print(best_val_loss_list)


def predict_k_fold(dict_args, pretrained_ckpt_path):
    print(f"Predict {dict_args['k_fold']}_fold checkpoints and generate results.")
    test_checkpoints = dict_args["test_from_checkpoints"]
    assert (len(test_checkpoints) == dict_args['k_fold'])
    arch = dict_args['arch']
    for k in range(dict_args['k_fold']):
        if arch == "frameatst":
            pretrained_module = FrameATSTPredModule(pretrained_ckpt_path,
                                                    finetune_layer=dict_args["finetune_layer"],
                                                    use_last=dict_args["use_last"],
                                                    drop_rate=0.0, attn_drop_rate=0.0)
        elif arch == 'mert':
            pretrained_module = MertPredModule(pretrained_ckpt_path)
        else:
            raise NotImplementedError(f"{arch} not supported!")
        pretrained_module.eval()  # 这里是新加的，原来是统一和train相同，但是在test模式下，只需要eval就可以，验证一下是否正确
        ckpt = test_checkpoints[k]
        print(f"Predict using {ckpt}......")
        run(dict_args, pretrained_module, k,
            save_path=os.path.join(dict_args["save_path"], f"fold_{k + 1}"),
            test_ckpt=ckpt)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    # os.environ["OMP_NUM_THREADS"] = str(8)
    main()
