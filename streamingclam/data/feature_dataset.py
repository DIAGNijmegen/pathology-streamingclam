import math
import pyvips
import pandas as pd
from pathlib import Path
import albumentationsxl as A

from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """ Dataset class for backbone feature extraction
    This dataset simply opens an image at a given read_level, applies cropping/padding to play nicely with streaming
    and then returns it. Optionally, masks can also be provided

    Augmentations are not applied, nor is there any class or label

    """

    def __init__(
            self,
            img_dir: Path | str,
            tile_size: int,
            img_size: int,
            read_level: int,
            mask_dir: Path | str | None = None,
            mask_suffix: str = "_tissue",
            variable_input_shapes: bool = False,
            tile_stride: int | None = None,
            network_output_stride: int = 1,
            output_dir: str | Path | None = None,
            filetype: str = ".tif"):

        self.img_dir = Path(img_dir)
        self.filetype = filetype
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_suffix = mask_suffix
        self.output_dir = Path(output_dir)

        self.read_level = read_level
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.network_output_stride = network_output_stride
        self.img_size = img_size

        self.variable_input_shapes = variable_input_shapes

        self.random_crop = A.RandomCrop(self.img_size, self.img_size, p=1.0)
        if not self.img_dir.exists():
            raise NotADirectoryError(f"Directory {self.img_dir} not found or doesn't exist")
        if not self.mask_dir.exists() and self.mask_dir is not None:
            raise NotADirectoryError(f"Directory {self.img_dir} not found or doesn't exist")

        self.images = list(self.img_dir.rglob(f"*{filetype}"))

        if self.output_dir is not None:
            self.filter_images()


    def get_img_pairs(self, idx):
        images = {"image": None, "mask":None}
        fnames = {"image_fname": None, "mask_fname" : None}

        img_path = self.images[idx]
        img_fname = img_path.stem
        image = pyvips.Image.new_from_file(str(img_path), page=self.read_level, access="sequential")
        images["image"] = image
        fnames["image_fname"] = img_fname

        if self.mask_dir is not None:
            mask_fname = img_fname + self.mask_suffix
            mask_path = self.mask_dir / Path(mask_fname).with_suffix(self.filetype)
            mask = pyvips.Image.new_from_file(mask_path)
            ratio = image.width / mask.width
            # could be done more efficiently here
            images["mask"] = mask.resize(ratio, kernel="nearest")  # Resize mask to img size
            fnames["mask_fname"] = mask_fname

        return images, fnames


    def filter_images(self):
        """ Filter images that have already been processed in output_dir
        """

        out_files_present = list(self.output_dir.rglob("*.pt"))
        out_file_names = [x.stem for x in out_files_present]
        already_processed = [self.images[0].parent / Path(x).with_suffix(self.filetype) for x in out_file_names]

        self.images = list(set(self.images) - set(already_processed))

    def __getitem__(self, idx):
        sample, fnames = self.get_img_pairs(idx)

        pad_to_tile_size = sample["image"].width < self.tile_size or sample["image"].height < self.tile_size
        # Get the resize op depending on image size
        resize_op = self.get_resize_op(pad_to_tile_size=pad_to_tile_size)
        sample = resize_op(**sample)

        # if after padding to the tile stride the image is bigger than img_size, crop it (upper bound on memory)
        if sample["image"].width * sample["image"].height > self.img_size ** 2:
            sample = self.random_crop(**sample)

        # Masks don't need to be really large for tissues, so scale them back
        if sample["mask"] is not None:
            # Resize to streamingclam output stride, with max pool kernel
            # Rescale between model max pool and pyvips might not exactly align, so calculate new scale values
            new_height = math.ceil(sample["mask"].height / self.network_output_stride)
            vscale = new_height / sample["mask"].height

            new_width = math.ceil(sample["mask"].width / self.network_output_stride)
            hscale = new_width / sample["mask"].width

            sample["mask"] = sample["mask"].resize(hscale, vscale=vscale, kernel="nearest")

        to_tensor = A.Compose([A.ToTensor(transpose_mask=True)], is_check_shapes=False)
        sample = to_tensor(**sample)

        # To ToTensor does not support cast to bool arrays yet, so do here
        if sample["mask"] is not None:
            sample["mask"] = sample["mask"] >= 1

        return sample, fnames

    def __len__(self):
        return len(self.images)

    def get_resize_op(self, pad_to_tile_size=False):
        if not self.variable_input_shapes:
            # Crop everything to specific image size if variable_input_shapes is off
            return A.Compose([A.CropOrPad(self.img_size, self.img_size, p=1.0)])

        # Pad images that are smaller than the tile size to the tile size
        if pad_to_tile_size:
            return A.Compose(
                [
                    A.PadIfNeeded(
                        min_width=self.tile_size, min_height=self.tile_size, value=[255, 255, 255], mask_value=[0, 0, 0]
                    )
                ]
            )

        # Images that are already larger than tile size should be padded to a multiple of tile_stride
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_width_divisor=self.tile_stride,
                    pad_height_divisor=self.tile_stride,
                    value=[255, 255, 255],
                    mask_value=[0, 0, 0],
                ),
            ]
        )

if __name__ == "__main__":
    root = Path("/data/pathology/projects/pathology-bigpicture-streamingclam")
    data_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images")
    mask_path = root / Path("data/breast/camelyon_packed_0.25mpp_tif/images_tissue_masks")

    dataset = FeatureDataset(
        img_dir=str(data_path),
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
    print("Creating dataset")
    sample, name = next(ds)
    print("retrieving image")
    image = sample["image"]
    mask = sample["mask"]
    print("image shape", image.shape)
    print("image dtype", image.dtype)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.show()
    print("mask shape", mask.shape)
    plt.imshow(mask.cpu().numpy())
    plt.show()
