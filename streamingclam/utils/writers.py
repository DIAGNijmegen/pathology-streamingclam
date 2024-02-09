import torch
import pyvips

import numpy as np
import pandas as pd

from pathlib import Path
from torch.nn.functional import softmax
from typing import Any, Optional, Sequence
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.callbacks import Callback


class AttentionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path | str, read_level: int, write_level: int, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.create_output_dirs()
        self.read_level = read_level
        self.write_level = write_level

    def create_output_dirs(self):
        if not self.output_dir.exists():
            print(f"Output directories for attentions don't exist, creating it at {self.output_dir}")
            self.output_dir.mkdir(parents=True)
            (self.output_dir / Path("attentions")).mkdir()
            (self.output_dir / Path("thumbnails")).mkdir()

    def transfer_to_device(self, data):
        if isinstance(data, tuple):
            data = (x.detach().cpu() for x in data)

        if isinstance(data, dict):
            for key, val in data.items():
                data[key] = val.detach().cpu() if isinstance(val, torch.Tensor) else val

        return data

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        batch = self.transfer_to_device(batch)

        attention = self.process_attention(batch).transpose(1,2,0)
        num_branches = attention.shape[-1]

        attention = pyvips.Image.new_from_array(attention)
        attention = attention.resize(256, kernel="linear")

        # Pyvips has weird problems if number of bands is not equal to 1 or 3.
        for band in range(num_branches):
            out_file_mask = self.output_dir / Path(batch["image_name"] + f"_attention_{band}").with_suffix(".tif")
            attention.write_to_file(out_file_mask, pyramid=True, compression="jpeg", tile=True)

        out_file = self.output_dir / Path(batch["image_name"]).with_suffix(".tif")
        image = pyvips.Image.new_from_array(batch["image"][0, ...].permute(1, 2, 0))
        image.write_to_file(out_file, pyramid=True, compression="jpeg", tile=True)

        print("test")

    def _process_attention_mask(self, batch, attention, num_branches):
        # Cast mask to booleans and back to 0,1
        mask = batch["mask"].numpy()
        mask = (mask > 0) * 1

        _, h, w = mask.shape

        # Get indices where mask > 0
        idx = np.where(mask.flatten())[0]

        masked_att = np.zeros((num_branches, h, w)).astype("uint8")
        # Flatten the array per branch
        masked_att = masked_att.reshape([num_branches, -1])

        # Fill the 0 array with attention values that are not masked for each branch
        masked_att[:, idx] = attention
        masked_att = masked_att.reshape((num_branches, h, w))

        return masked_att

    def _process_attention_nomask(self, batch, attention, num_branches):
        # Get the dimensions of the feature map just before the attention head
        fmap_dims = batch["image"].shape[-2:] / batch["output_stride"].cpu().numpy()

        fmap_dims = [int(x) for x in fmap_dims]

        fmap_dims.append(num_branches)

        attention = attention.reshape(fmap_dims).numpy()
        return attention

    def process_attention(self, batch):
        attention = softmax(batch["A_raw"], dim=1) * 255  # (uint8 viewing)
        attention = attention.to(torch.uint8)
        num_branches = attention.shape[0]

        if "mask" in batch.keys():
            attention = self._process_attention_mask(batch, attention, num_branches)
        else:
            attention = self._process_attention_nomask(batch, attention, num_branches)

        return attention

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Create a csv with CLAM-like results, e.g. logits, y_prob, y_hat, acc, auc, etc..
        pass

class TestPredictionWriter(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.create_output_dirs()

    def create_output_dirs(self):
        if not self.output_dir.exists():
            print(f"Output directories for attentions don't exist, creating it at {self.output_dir}")
            self.output_dir.mkdir(parents=True)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        test_df = pd.DataFrame(pl_module.test_outputs)
        test_df.to_csv(str(self.output_dir / Path("test.csv")), index=False)




