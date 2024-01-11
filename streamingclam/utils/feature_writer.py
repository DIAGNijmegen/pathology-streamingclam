import torch
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter

class FeatureWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        images = prediction[0]
        names = prediction[1]

        out_file = Path(self.output_dir) / Path(names["image_fname"][0]).with_suffix(".pt")

        torch.save(images, out_file)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        print("HI")
        #torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))