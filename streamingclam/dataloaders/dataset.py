import pyvips
import math
import torch

import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset


class StreamingClassificationDataset(Dataset):
    def __init__(
        self,
        img_dir,
        csv_file,
        tile_size,
        img_size,
        transform,
        mask_dir=None,
        mask_suffix="_tissue",
        variable_input_shapes=False,
        tile_delta=None,
        network_output_stride=1,
        filetype=".tif",
        read_level=0,
        *args,
        **kwargs,
    ):
        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_delta = tile_delta
        self.network_output_stride = network_output_stride
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform

        self.classification_frame = pd.read_csv(csv_file)

        # Will be populated in check_csv function
        self.data_paths = {"images": [], "masks": [], "labels": []}

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory {self.img_dir} not found or doesn't exist")
        self.check_csv()

        self.labels = self.data_paths["labels"]

    def check_csv(self):
        """Check if entries in csv file exist"""
        included = {"images": [], "masks": [], "labels": []} if self.mask_dir else {"images": [], "labels": []}
        for i in range(len(self)):
            images, label = self.get_img_path(i)  #

            # Files can be just images, but also image, mask
            for file in images:
                if not file.exists():
                    print(f"WARNING {file} not found, excluded both image and mask (if present)!")
                    continue

            included["images"].append(images[0])
            included["labels"].append(label)

            if self.mask_dir:
                included["masks"].append(images[1])

        self.data_paths = included

    def get_img_path(self, idx):
        img_fname = self.classification_frame.iloc[idx, 0]
        label = self.classification_frame.iloc[idx, 1]

        img_path = self.img_dir / Path(img_fname).with_suffix(self.filetype)

        if self.mask_dir:
            mask_path = self.mask_dir / Path(img_fname + self.mask_suffix).with_suffix(self.filetype)
            return [img_path, mask_path], label

        return [img_path], label

    def __getitem__(self, idx):
        img_fname = str(self.data_paths["images"][idx])
        label = int(self.data_paths["labels"][idx])

        image = pyvips.Image.new_from_file(img_fname, page=self.read_level)
        sample = {"image": image}

        if self.mask_dir:
            mask_fname = str(self.data_paths["masks"][idx])
            mask = pyvips.Image.new_from_file(mask_fname)
            ratio = image.width / mask.width
            sample["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size

        if self.transform:
            # print("applying transforms")
            sample = self.transform(**sample)

        # Output of transforms are uint8 images in the range [0,255]
        normalize = T.Compose(
            [
                T.PadIfNeeded(
                    pad_height_divisor=self.tile_delta,
                    pad_width_divisor=self.tile_delta,
                    min_height=None,
                    min_width=None,
                    value=[255, 255, 255],
                    mask_value=[0, 0, 0],
                ),
                T.ToDtype("float", scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        sample = normalize(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        # TODO: Make transformation that operates on mask alone to resize
        if self.mask_dir:
            # Resize to streamingclam output stride, with max pool kernel
            sample["mask"] = sample["mask"].resize(1 / self.network_output_stride, kernel="nearest")

        to_tensor = T.Compose([T.ToTensor(transpose_mask=True)], is_check_shapes=False)
        sample = to_tensor(**sample)

        if self.mask_dir:
            sample["mask"] = sample["mask"] >= 1
            return sample["image"], sample["mask"], torch.tensor(label)

        return sample["image"], torch.tensor(label)

    def __len__(self):
        return len(self.classification_frame)


if __name__ == "__main__":
    root = Path("/data/pathology/projects/pathology-bigpicture-streamingclam")
    data_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images")
    mask_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks")
    csv_file = root / Path("streaming_experiments/camelyon/data_splits/train_0.csv")

    dataset = StreamingClassificationDataset(
        img_dir=str(data_path),
        csv_file=str(csv_file),
        tile_size=1600,
        img_size=3200,
        transform=[],
        mask_dir=mask_path,
        mask_suffix="_tissue",
        variable_input_shapes=True,
        tile_delta=680,
        filetype=".tif",
        read_level=1,
    )

    for x in dataset:
        print(x[0].shape, x[1].shape, x[2])
