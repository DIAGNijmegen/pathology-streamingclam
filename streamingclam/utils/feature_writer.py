import torch
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter

class FeatureWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        print("hi")
        #torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        print("HI")
        #torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))