import pyvips
import logging
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt


class Heatmap:
    def __init__(self, width: int, height: int, bands: int = 1, write_level: int = 0):
        self.width = width
        self.height = height
        self.bands = bands

        self.canvas = None
        self.mini_canvas = None
        self.scores = None
        self.write_level = write_level

    def create_mini_heatmap(
        self,
        coords: np.array,
        scores: np.array,
        patch_size: int,
        step_size: int | None = None,
    ):
        """Create the smallest possible heatmap, i.e. such that 1 patch equals 1 pixel.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the patches.
        scores : np.ndarray
            Scores corresponding to the patches.
        patch_size : int
            Size of the patch.
        step_size : int, optional
            Step size for the patches. Defaults to None.

        Returns
        -------
        np.ndarray
            Mini heatmap array.
        """

        # Assuming patch_size=step_size
        # Make the smallest canvas possible, such that 1 score is 1 pixel
        mini_canvas = np.zeros(
            (
                int(np.ceil(self.height / patch_size)),
                int(np.ceil(self.width / patch_size)),
                scores.shape[0],
            )
        )

        coords = (coords / patch_size).astype("int")
        for coord, score in zip(coords, scores.transpose()):
            x, y = int(coord[0]), int(coord[1])
            mini_canvas[y, x, :] = score

        self.patch_size = patch_size
        self.mini_canvas = mini_canvas
        self.scores = scores

    def resize_mini_canvas(self, scale):
        if self.mini_canvas is None:
            raise ValueError("mini_canvas is None, please run create_mini_heatmap before using this function")

        canvas = pyvips.Image.new_from_array(self.mini_canvas)
        canvas = canvas.resize(scale, kernel="cubic")
        return canvas

    def save_png_thumbnail(self, save_path: str | Path, titles: list | None = None,
                           vmin=None, vmax=None, hist_bins: int = 50):
        """
        Save a PNG thumbnail of the heatmap. Creates subplots if there are multiple bands.
        Also adds a second row with histograms of the values being plotted.

        Parameters
        ----------
        save_path : str | Path
            Complete path where the PNG thumbnail will be saved.
        titles : list | None
            List of titlenames, must be of the same length as the number of channels C in mini_canvas
        vmin, vmax : float | None
            Color limits for the images; if None, computed per-band.
        hist_bins : int
            Number of bins for the histograms.
        """
        if self.mini_canvas is None:
            raise ValueError("mini_canvas is None, please run create_mini_heatmap before using this function")

        num_bands = self.mini_canvas.shape[2]
        if titles:
            assert len(titles) == num_bands

        # 2 rows: images on top, histograms below
        fig, axes = plt.subplots(
            2, num_bands,
            figsize=(5 * num_bands, 6.5),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35, "wspace": 0.3}
        )
        # Ensure axes is 2D: shape (2, num_bands)
        if num_bands == 1:
            axes = np.array(axes).reshape(2, 1)

        for i in range(num_bands):
            img_ax = axes[0, i]
            hist_ax = axes[1, i]

            data = np.asarray(self.mini_canvas[:, :, i])
            # Range used for both image and histogram (keeps them consistent)
            dmin = np.nanmin(data) if vmin is None else vmin
            dmax = np.nanmax(data) if vmax is None else vmax

            im = img_ax.imshow(data, cmap="coolwarm", vmin=dmin, vmax=dmax)
            img_ax.set_title(titles[i] if titles else f"class_{i}")
            img_ax.axis("off")

            # Colorbar for each image subplot
            cbar = fig.colorbar(im, ax=img_ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            # Histogram on the second row
            data = np.asarray(self.scores[...,i])
            flat = data[np.isfinite(data)].ravel()
            if flat.size > 0:
                hist_ax.hist(flat, bins=hist_bins, range=(dmin, dmax))
            hist_ax.set_xlim(dmin, dmax)
            hist_ax.set_ylabel("count", fontsize=8)
            hist_ax.set_xlabel("value", fontsize=8)
            hist_ax.tick_params(axis='both', labelsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_lowest_downsample_factor(self):
        scale_orig = self.count_divisibility_by_2([self.width, self.height])  # original image divisibility by 2

        height, width = self.mini_canvas.shape[0:2]
        # mini canvas is a padded version of the original image, so that each patch is at least patch_size

        downsample_factor = min(scale_orig, self.patch_size // 2)

        return downsample_factor

    def get_scale(self) -> int:
        """
        Retrieves the scaling factor for the resizing operation. It takes into account the user-specified write_level
        and a lower bound on the lowest possible resolution we can possibly write

        Returns
        -------

        """
        max_scale = self.get_lowest_downsample_factor()  # Lower bound on scaling, we cannot go smaller than this

        check = max_scale - self.write_level
        if check < 0:
            logging.warning(
                f"Cannot write at write_level {self.write_level}, because level {check} is the lowest possible."
                f"Instead, the result will be written at level {check}"
            )
            return max_scale

        return self.write_level


    def save_tif_heatmap(self, save_paths: list[str] | list[Path]):
        """
        Save scores as tif files. Each individual band will be saved as a single tif file
        This is to ensure ASAP can read it

        Parameters
        ----------
        save_paths : list[str] | list[Path]
            List of strings of list of Paths with the absolute paths where the files should be saved
            The list length should be equal to the number of channels C in the mini canvas
        """

        if self.mini_canvas is None:
            raise ValueError("mini_canvas is None, please run create_mini_heatmap before using this function")

        assert len(save_paths) == self.mini_canvas.shape[2]

        scale = self.get_scale()

        canvas = self.resize_mini_canvas(scale=self.patch_size // 2**scale)

        # mini_map may now be larger due to patches going over the image border. Correct this with a crop
        canvas = canvas.crop(0, 0, self.width // 2**scale, self.height // 2**scale)
        canvas = canvas.bandsplit()
        for i, band in enumerate(canvas):
            temp = band.copy(interpretation="b-w")
            temp = temp.scaleimage()
            temp.write_to_file(str(save_paths[i]), bigtiff=True, pyramid=True, tile=True, compression="lzw", xres=2000, yres=2000, resunit="cm")

    @staticmethod
    def count_divisibility_by_2(numbers: int | tuple | list):
        def count_divisions(n):
            count = 0
            while n % 2 == 0:
                n //= 2
                count += 1
            return count

        if isinstance(numbers, int):
            return count_divisions(numbers)

        # Apply the counting function to each number and return the results in a list
        counts = [count_divisions(num) for num in numbers]

        # Find the minimum count
        min_count = min(counts)

        # Return the number with the minimum count (or one of them if there are multiple)
        return min_count
