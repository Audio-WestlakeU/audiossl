from audiossl.methods.atstframe.downstream.utils_as_strong.model_as_strong import FineTuningPLModule
from audiossl.methods.atstframe.downstream.comparison_models.clip_atst_module import ATSTPredModule
from audiossl.methods.atstframe.downstream.comparison_models.frame_atst_module import FrameATSTPredModule

class DistillTeacherModule(FineTuningPLModule):
    def __init__(self,
                 mode="clip"):
        if mode == "clip":
            encoder = ATSTPredModule("./comparison_models/ckpts/clip_atst.ckpt")
        elif mode == "frame":
            encoder = FrameATSTPredModule("./comparison_models/ckpts/frame_atst.ckpt")
        super().__init__(encoder=encoder, learning_rate=None, max_epochs=None, warmup_epochs=None, num_labels=407, niter_per_epoch=None)

    def forward(self, batch):
        self.encoder.eval()
        x, labels = self.encoder(batch)
        strong_pred = self.head(x)
        return strong_pred