
import pytorch_lightning as pl
from pytorch_lightning import LightningModule,Trainer
import math
import os

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = "{}_{}_{}.ckpt".format(self.prefix,epoch,global_step)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class CheckpointEveryNEpochs(pl.Callback):
    """
    Save a checkpoint every N epochs, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_epoch_frequency,
        prefix="N-epoch-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_epoch_frequency: how often to save in epochs
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_epoch_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = "{}_{}.ckpt".format(self.prefix,epoch)
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class WeightDecayCosineScheduler(pl.Callback):
    """
    """

    def __init__(
        self,
        wd_init,
        wd_final
    ):
        """
        Args:
        """
        self.wd_init = wd_init
        self.wd_final = wd_final

    def cosine(self,step,max_steps,init,final):
        wd = final - (final - init) * (math.cos(math.pi * step / max_steps) + 1) / 2
        return wd

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs ,
        batch ,
        batch_idx: int,
        unused: int = 0,
    ) -> None:

        epoch = trainer.current_epoch
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        wd = self.cosine(pl_module.global_step,max_steps,self.wd_init,self.wd_final)
        for i, param_group in enumerate(trainer.optimizers[0].param_groups):
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd
        
        pl_module.log("wd",wd,prog_bar=True,logger=True)
        pl_module.log("lr",param_group["lr"],prog_bar=True,logger=True)
