import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from pathlib import Path
from typing import Any, Optional, Sequence

from lightning.pytorch.callbacks import BasePredictionWriter
from streamingclam.utils.heatmap import Heatmap


class StreamingHeatmapWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path, write_interval, write_level: int = 0, overwrite: bool = False):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.heatmap_dir = None
        self.write_level = write_level
        self.overwrite = overwrite
        self.skip = False

    def create_output_dirs(self, fname: str):
        # these class variables changes with each new image in the test set batch being processed
        self.heatmap_dir = self.output_dir / Path("heatmaps") / Path(fname)

        if not self.overwrite and self.heatmap_dir.exists():
            self.skip = True
            print(f"Heatmap dir {self.heatmap_dir} exists and overwrite is {self.overwrite}, skipping")

        if not self.heatmap_dir.exists():
            self.heatmap_dir.mkdir(parents=True, exist_ok=True)

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
        logits, Y_prob, Y_hat, A_raw = prediction

        metadata = batch.metadata
        mask = batch.mask.detach().cpu().numpy()[0, ...]
        output_stride = trainer.predict_dataloaders.dataset.network_output_stride
        metadata["coords_level"] = self.get_coords_from_mask(mask) * output_stride
        self.process_wsi(metadata, logits, Y_prob, Y_hat, A_raw, batch.label)
        print(f'Done processing {metadata["filename"]}')

    def process_wsi(self, metadata: dict, logits, Y_prob, Y_hat, A_raw, label: torch.Tensor):
        self.create_output_dirs(fname=metadata["filename"])

        # skip if result exists, writing with pyvips can take very long.
        if self.skip:
            self.skip = False
            return

        # Normalized attention for visualization
        A_norm = (A_raw - A_raw.min()) / (A_raw.max() - A_raw.min())
        hmap = self.create_mini_canvas(metadata, A_norm.detach().cpu().float().numpy())
        self.create_and_save_heatmaps(hmap, [f"class_{i}" for i in range(A_norm.shape[0])])
        self.create_and_save_thumbnails(
            hmap, f"Attention", [f"class_{i}" for i in range(A_norm.shape[0])], vmin=0, vmax=1
        )

        self.write_slide_statistics(Y_prob, Y_hat, label)
        torch.save(metadata, self.heatmap_dir / Path("metadata.pt"))

    def create_mini_canvas(self, metadata: dict, scores: np.ndarray):
        hmap = Heatmap(metadata["width_level"], metadata["height_level"], scores.shape[0], write_level=self.write_level)
        hmap.create_mini_heatmap(coords=metadata["coords_level"], scores=scores, patch_size=metadata["patch_size"])
        return hmap

    def create_and_save_thumbnails(
        self, hmap: Heatmap, title: str, bandnames: list | None = None, vmin: float = None, vmax: float = None
    ):
        thumbnail_path = Path(self.heatmap_dir) / f"{title}.png"
        hmap.save_png_thumbnail(save_path=thumbnail_path, titles=bandnames, vmin=vmin, vmax=vmax)

    def create_and_save_heatmaps(self, hmap: Heatmap, bandnames: list | None = None):
        bandnames = [x + ".tif" for x in bandnames]
        save_paths = [self.heatmap_dir / x for x in bandnames]
        hmap.save_tif_heatmap(save_paths=save_paths)

    def write_slide_statistics(self, prob, pred, label):
        df = pd.DataFrame()
        df["label"] = [label.detach().cpu().squeeze().float().numpy()]
        df["probs"] = [prob.detach().cpu().float().numpy().squeeze()]
        df["y_hat"] = pred.detach().cpu().float().numpy()

        # Split the list into separate columns
        split_cols = pd.DataFrame(df["probs"].apply(lambda x: [val for val in x]).tolist())

        # Rename columns to p_0, p_1, etc.
        split_cols.columns = [f"p_{i}" for i in range(split_cols.shape[1])]

        # Join with the original DataFrame (optional, depending on your needs)
        df = df.drop(columns=["probs"]).join(split_cols)
        df.to_csv(self.heatmap_dir / Path("slide_stats.csv"), index=False)

    @staticmethod
    def get_coords_from_mask(mask) -> np.ndarray:
        """

        Parameters
        ----------
        mask : np.ndarray
            A (H, W) mask

        Returns
        -------
        coords: np.ndarray
            A [N, 2] list of (x,y) coordinates of which pixels are non-zero
        """

        # Ensure the mask after squeezing is a 2D array
        if mask.ndim != 2:
            raise ValueError("Input mask must be of shape (H, W)")

        # Get the y, x indices where the mask is non-zero
        y_indices, x_indices = np.where(mask > 0)

        # Stack the x and y indices into a [N, 2] array
        coords = np.column_stack((x_indices, y_indices))

        return coords
