import math
import torch
import pyvips
import warnings
import logging

from pathlib import Path
from functools import partial

import pandas as pd
import albumentationsxl as A
from torchvision.transforms import Normalize


class GenericStreamingDataset(torch.utils.data.Dataset):
    """
    Helper class that handles most of the input handling and path checking
    This class concretely checks for the existence and correct specification of the image directories for images and
    masks, and the existence of the csv path with the filename column, and label column.

    If the image, mask, or csv paths do not exist, a NotADirectoryError is thrown
    If any of the files specified in the csv cannot be found, a FileNotFoundError will be thrown, along with all
    the image/mask paths that could not be resolved


    """

    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        mask_dir=None,
        image_col: str = "slide",
        label_col: str = "label",
        mask_suffix: str = "_tissue",
        filetype: str = ".tif",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_suffix = mask_suffix
        self.filetype = filetype

        self.image_col = image_col
        self.label_col = label_col

        self.check_input_paths()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.labels = self.df[self.label_col]  # Needed for the sampler, which expects a labels attribute

        self.check_images()

    def check_input_paths(self):
        if not self.image_dir.exists():
            raise NotADirectoryError(f"image dir {self.image_dir} not accessible or does not exist")

        if self.mask_dir:
            if not self.mask_dir.exists():
                raise NotADirectoryError(f"mask dir {self.mask_dir} not accessible or does not exist")

    def check_images(self):
        self.get_image_paths()

        missing_images = self.get_missing_images("image_path")
        missing_masks = self.get_missing_images("mask_path") if self.mask_dir else []

        if missing_images or missing_masks:
            warnings.warn(f"Found images in the csv, that are (possibly) not in the data folder")
            for x in missing_images:
                print(x)

            for x in missing_masks:
                print(x)

            raise FileNotFoundError("Several images/masks could not be found")

        logging.info("All images/masks found")

    def get_missing_images(self, colname: str):
        missing_paths = []
        for image_path in self.df[colname]:
            missing_paths.append(image_path) if not image_path.exists() else None

        return missing_paths

    def get_image_paths(self):
        self.df["image_path"] = self.df[self.image_col].apply(lambda x: Path(self.image_dir) / Path(x + self.filetype))

        if self.mask_dir:
            self.df["mask_path"] = self.df[self.image_col].apply(
                lambda x: Path(self.mask_dir) / Path(x + self.mask_suffix + self.filetype)
            )

    def __len__(self):
        return len(self.df)


class StreamingDataset(GenericStreamingDataset):
    """
    Dataset generator for Streaming models. By default, this class performs resizing/padding/cropping such that it works
    with a given streaming model, provided by tile_size, network_output_stride, and tile_stride.

    Parameters
    ----------
    image_dir : str | Path
        Absolute path to the directory that contains the images.
        The extension to look for images in a glob expression by default is ".tif", and can be changed with filetype
    csv_path : str | Path
        Absolute path to the csv with the image names and their corresponding labels.
    tile_size : int
        The tile size of the streaming network
    image_size : int
        The (maximum) size of the image. If variable_input_shapes is set to True, then all input images will either
        be cropped or padded to this size in both spatial dimensions. If False, then this variable will be used as
        a maximum, and any image greater than image_size in either dimension will be cropped to image_size to prevent
        memory overflow
    read_level: int
        The level from which to read the image. 0 Equals the highest resolution.
    transform: A.BaseCompose | None
        An albumentationsXL compose object with additional augmentations that can be used during training
    mask_dir: str | Path | None
            Directory that contains the images. The extension by default is ".tif", and can be changed with filetype.
            Default is set to None
    mask_suffix: str
        Any additional suffix that might be added to the filename, e.g. <IMAGE_ID_tissue.tif>, where _tissue is the
        mask_suffix
    variable_input_shapes: bool
        Whether images should always be padded/cropped to image_size. If set to false, input images can be substantially
        smaller if their original dimensions are << image_size. This can help speed up training, and use less memory.
    tile_stride: int | None
        The tile stride of the streaming network. This is needed to play nicely with streaming
    network_output_stride: int
        The output stride of the streaming network of the final layer. Most CNN models e.g. ResNet will downsample
        image dimensions by a factor of 32. So any image of say 320x320 will be downsampled to 32x32xC. This variable
        is needed to correctly resize masks
    filetype: str
        The file extension of the image/masks. Default is ".tif"
    heatmap_mode: bool
        In this mode, no augmentations are allowed, and crop/padding operations are configured to be
        deterministic and their position parameters changed to north-west (top left position), which means all padding
         and crop operations will only add/remove pixels on the bottom and right of the image. Default is False
    """

    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        tile_size: int,
        image_size: int,
        read_level: int,
        transform: A.BaseCompose | None = None,
        normalize: bool = True,
        mask_dir: Path | str | None = None,
        mask_suffix: str = "_tissue",
        variable_input_shapes: bool = False,
        tile_stride: int | None = None,
        network_output_stride: int = 1,
        filetype: str = ".tif",
        heatmap_mode: bool = False,
    ):
        super().__init__(image_dir, csv_path, mask_dir, mask_suffix=mask_suffix, filetype=filetype)

        if heatmap_mode and transform:
            raise ValueError(
                f"transform cannot be used when heatmap_mode is set to True, but found {heatmap_mode}, {transform}"
            )

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.network_output_stride = network_output_stride
        self.img_size = image_size
        self.heatmap_mode = heatmap_mode
        self.normalize = normalize

        self.variable_input_shapes = variable_input_shapes
        self.transform = transform
        self.metadata = {}

        # Define where crop/pads take place in the image. For attention, everything is on the bottom/right
        self.direction = "north-west" if self.heatmap_mode else "centre"

        # Some useful transforms we need to use throughout the code
        self.to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
        self.normalization = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_op = (
            A.RandomCrop(self.img_size, self.img_size, p=1.0)
            if not heatmap_mode
            else A.Crop(x_min=0, y_min=0, x_max=self.img_size, y_max=self.img_size, p=0.0)
        )

    def __getitem__(self, item):
        self.metadata = {}
        image_path, mask_path, label, filename = self.get_item_from_df(item)
        self.metadata.update({"image_path": image_path})
        self.metadata.update({"filename": filename})

        images = self.gather_image_pairs(image_path, mask_path)  # Get image/mask pair (if available) from paths
        images = self.transform(**images) if self.transform else images  # Any augmentations
        images = self.crop_or_pad_image(images)  # crop/pad images after augmentations to a suitable size for streaming

        # if after padding to the tile stride the image is bigger than img_size, crop it (upper bound on memory)
        self.metadata["is_cropped"] = False  # Needed for heatmaps. We only write heatmaps that fit the dimensions
        if images["image"].width * images["image"].height > self.img_size**2:
            images = self.crop_op(**images)
            self.metadata["is_cropped"] = True

        images = self.resize_mask(images) if "mask" in images.keys() else images  # Masks are resized to output stride
        images = self.convert_to_tensor(images)  # Convert to pytorch tensors for training

        if self.normalize: # normalizing in dataloader will convert to float, which can be expensive for large images
            images["image"] = self.normalization(images["image"] / 255.0)

        self.metadata.update(
            {"patch_size": self.network_output_stride, "tile_stride": self.tile_stride, "level": self.read_level}
        )
        return {
            "image": images["image"],
            "mask": images["mask"] if "mask" in images.keys() else None,
            "label": torch.as_tensor([label]),
            "metadata": self.metadata,
        }

    def get_item_from_df(self, item):
        if not self.mask_dir:
            return self.df["image_path"].iloc[item], None, self.df["label"].iloc[item], self.df["slide"].iloc[item]
        return tuple(self.df[["image_path", "mask_path", "label", "slide"]].iloc[item])

    def gather_image_pairs(self, image_path: Path | str, mask_path: Path | str | None = None):
        images = {"image": pyvips.Image.new_from_file(image_path, page=self.read_level, access="sequential")}
        self.metadata.update({"width_level": images["image"].width, "height_level": images["image"].height})

        if mask_path:
            mask = pyvips.Image.new_from_file(mask_path, access="sequential")
            ratio = images["image"].width / mask.width
            images["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size
            self.metadata.update({"width_level_mask": images["mask"].width, "height_level_mask": images["mask"].height})
        return images

    def crop_or_pad_image(self, images: dict):
        pad_to_tile_size = images["image"].width < self.tile_size or images["image"].height < self.tile_size
        resize_op = self.get_crop_or_pad_op(pad_to_tile_size=pad_to_tile_size)  # Get operation depending on image size
        images = resize_op(**images)
        return images


    def resize_mask(self, images: dict):
        # Resize to streamingclam output stride, with max pool kernel
        # Rescale between model max pool and pyvips might not exactly align, so calculate new scale values
        new_height = math.ceil(images["mask"].height / self.network_output_stride)
        new_width = math.ceil(images["mask"].width / self.network_output_stride)

        hscale, vscale = new_width / images["mask"].width, new_height / images["mask"].height
        images["mask"] = images["mask"].resize(hscale, vscale=vscale, kernel="nearest")
        return images

    def convert_to_tensor(self, images: dict):
        images = self.to_tensor(**images)

        if "mask" in images.keys():
            images["mask"] = images["mask"] >= 1  # To ToTensor does not support cast to bool arrays yet, so do here
        return images

    def get_crop_or_pad_op(self, pad_to_tile_size=False):
        if not self.variable_input_shapes:
            return A.Compose([A.CropOrPad(self.img_size, self.img_size, direction=self.direction, p=1.0)])

        if pad_to_tile_size:  # Pad images that are smaller than the tile size to the tile size
            return self.pad_to_tile_size_op()

        pad_op_div = partial(
            A.PadIfNeeded,
            min_height=None,
            min_width=None,
            value=[255, 255, 255],
            mask_value=[0, 0, 0],
            position=self.direction,
        )
        # Images that are already larger than tile size should be padded to a multiple of tile_stride
        return A.Compose([pad_op_div(pad_width_divisor=self.tile_stride, pad_height_divisor=self.tile_stride)])

    def pad_to_tile_size_op(self):
        pad_op_min = partial(A.PadIfNeeded, value=[255, 255, 255], mask_value=[0, 0, 0], position=self.direction)
        pad_op_div = partial(
            A.PadIfNeeded,
            min_height=None,
            min_width=None,
            value=[255, 255, 255],
            mask_value=[0, 0, 0],
            position=self.direction,
        )

        pad_1 = pad_op_min(min_width=self.tile_size, min_height=self.tile_size)  # image must be at least tile_size
        # if one dimension is > tile size, make it a multiple of the output stride
        pad_2 = pad_op_div(pad_width_divisor=self.network_output_stride, pad_height_divisor=self.network_output_stride)
        return A.Compose([pad_1, pad_2])
