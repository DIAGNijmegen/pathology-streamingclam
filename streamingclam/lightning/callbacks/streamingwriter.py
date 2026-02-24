import torch
import numpy as np
import lightning.pytorch as pl

from pathlib import Path
from typing import Any, Optional, Sequence

from src.mixmil.utils.callbacks.milwriter import MILHeatmapWriter


class StreamingHeatmapWriter(MILHeatmapWriter):
    def __init__(self, output_dir: str | Path, write_interval, write_level: int = 0, overwrite: bool = False):
        super().__init__(output_dir, write_interval, write_level, overwrite)

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
        self.n_samples = pl_module.n_samples
        u, t, _w = prediction

        metadata = batch.metadata
        mask = batch.mask.detach().cpu().numpy()[0, ...]
        output_stride = trainer.predict_dataloaders.dataset.network_output_stride
        metadata["coords_level"] = self.get_coords_from_mask(mask) * output_stride
        self.process_wsi(metadata, t, _w, batch.label)
        print(f'Done processing {metadata["filename"]}')

    def process_wsi(self, metadata: dict, t: torch.Tensor, _w: torch.Tensor, label: torch.Tensor):
        self.create_output_dirs(fname=metadata["filename"])

        # skip if result exists, writing with pyvips can take very long.
        if self.skip:
            self.skip = False
            return

        u_wsi = t * _w.softmax(0)  # Spatial dimensions preserved
        u_mean, t_mean, _w_mean_norm = self.calculate_sample_means(u_wsi, t, _w)

        scores = [t]
        means = [t_mean, _w_mean_norm]
        bandnames = ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty", "JS_divergence"]
        names = ["t", "_w"]

        # Write uncertainty maps
        for score, name in zip(scores, names):
            entropy_calc = self.calculate_uncertainties(score)
            jsd_calc = self.calculate_jsd(score)
            thumb_uncertainties = np.vstack([np.vstack(entropy_calc), jsd_calc]).T

            heatmap_bandnames = [name + "_" + x for x in bandnames]
            hmap = self.create_mini_canvas(metadata, thumb_uncertainties)
            # self.create_and_save_heatmaps(hmap, heatmap_bandnames)
            self.create_and_save_thumbnails(hmap, f"{name}_uncertainties", bandnames, vmin=0, vmax=1)

        names = ["_w", "t", "t_logit", "u_logit", "attr"]

        z = _w.mean(2)
        # Write average probability/attention scores
        means = [z, t.softmax(1).mean(2), t.mean(2), u_wsi.mean(2)]

        for mean, name in zip(means, names):
            vmin, vmax = (0, 1) if name == "t" else (None, None)

            hmap = self.create_mini_canvas(metadata, mean.detach().cpu().float().numpy())
            # self.create_and_save_heatmaps(hmap, [f"class_{i}" for i in range(t_mean.shape[1])])
            self.create_and_save_thumbnails(
                hmap, f"{name}_mean", [f"class_{i}" for i in range(t_mean.shape[1])], vmin=vmin, vmax=vmax
            )

        self.write_slide_statistics(u_wsi, label)
        self.write_predictions(t.mean(2), label)
        torch.save(metadata, self.heatmap_dir / Path("metadata.pt"))

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
