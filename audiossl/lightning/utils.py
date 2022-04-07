from tkinter import W
import pytorch_lightning


from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule,LightningModule

class EmbeddingExtractor:
    def __init__(self,
                 module:LightningModule,
                 nproc:int=1
                ):
        self.trainer = Trainer(
                            strategy="ddp_find_unused_parameters_false",
                            sync_batchnorm=True,
                            logger=False,
                            gpus=nproc,
                            #profiler="simple",
                            #max_epochs=1,
                            )
        self.module = module
    def extract(self,dataloader):
        return self.trainer.predict(self.module,dataloader)


