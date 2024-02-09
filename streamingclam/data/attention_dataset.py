
import math
import pandas as pd
from pathlib import Path
from streamingclam.data.dataset import StreamingClassificationDataset
import albumentationsxl as A


class AttentionDataset(StreamingClassificationDataset):
    def __init__(
        self,
        img_dir: Path | str,
        image_df: pd.DataFrame,
        tile_size: int,
        img_size: int,
        read_level: int,
        mask_dir: Path | str | None = None,
        mask_suffix: str = "_tissue",
        variable_input_shapes: bool = False,
        tile_stride: int | None = None,
        network_output_stride: int = 1,
        filetype: str = ".tif",
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.network_output_stride = network_output_stride
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes

        self.classification_frame = image_df

        # Will be populated in check_csv function
        self.data_paths = {"images": [], "masks": [], "labels": []}

        self.random_crop = A.RandomCrop(self.img_size, self.img_size, p=1.0)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory {self.img_dir} not found or doesn't exist")
        self.check_csv()

        self.labels = self.data_paths["labels"]

    def __getitem__(self, idx):
        sample, label, img_fname = self.get_img_pairs(idx)
        height, width = sample["image"].height, sample["image"].width

        pad_to_tile_size = sample["image"].width < self.tile_size or sample["image"].height < self.tile_size
        # Get the resize op depending on image size
        resize_op = self.get_resize_op(pad_to_tile_size=pad_to_tile_size)
        sample = resize_op(**sample)

        # if after padding to the tile stride the image is bigger than img_size, crop it (upper bound on memory)
        if sample["image"].width * sample["image"].height > self.img_size**2:
            sample = self.random_crop(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        if "mask" in sample.keys():
            new_height = math.ceil(sample["mask"].height / self.network_output_stride)
            vscale = new_height / sample["mask"].height

            new_width = math.ceil(sample["mask"].width / self.network_output_stride)
            hscale = new_width / sample["mask"].width

            sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")

        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
        sample = to_tensor(**sample)

        # To ToTensor does not support cast to bool arrays yet, so do here
        if "mask" in sample.keys():
            sample["mask"] = sample["mask"] >= 1

        output = {
            "image_name": Path(img_fname).stem,
            "image_height": height,
            "image_width": width,
            "output_stride": self.network_output_stride,
            "label": label,
        }
        output.update(sample)

        return output


if __name__ == "__main__":
    root = Path("/data/pathology/projects/pathology-bigpicture-streamingclam")
    data_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images")
    mask_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks")
    csv_file = root / Path("streaming_experiments/camelyon/data_splits/train_0.csv")

    df = pd.read_csv(csv_file)
    dataset = AttentionDataset(
        img_dir=str(data_path),
        csv_file=df,
        tile_size=1600,
        img_size=1600,
        mask_dir=mask_path,
        mask_suffix="_tissue",
        variable_input_shapes=False,
        tile_stride=680,
        filetype=".tif",
        read_level=5,
    )

    import matplotlib.pyplot as plt

    ds = iter(dataset)
    check = next(ds)

    print(check.keys())
