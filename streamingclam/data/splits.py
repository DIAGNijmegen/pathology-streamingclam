import lightning as L
import albumentationsxl as A
from torch.utils.data import DataLoader
from pathlib import Path

from streamingclam.data.sampler import weighted_sampler
from streamingclam.data.dataset import StreamingClassificationDataset


class StreamingCLAMDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_dir: Path | str,
        level: int,
        tile_size: int,
        tile_stride: int,
        network_output_stride: int,
        train_csv_path: str | Path | None = None,
        val_csv_path: str | Path | None = None,
        test_csv_path: str | Path | None = None,
        tissue_mask_dir: str | Path | None = None,
        mask_suffix: str | None = None,
        image_size: int | None = None,
        variable_input_shapes: bool = True,
        copy_to_gpu: bool = False,
        num_workers: int = 2,
        transform: A.BaseCompose | None = None,
        verbose: bool = True,
        filetype: str = ".tif"
    ):
        super().__init__()
        self.image_dir = image_dir

        # Only for training, during testing only the data dir is used
        self.train_csv_path = Path(train_csv_path) if train_csv_path else train_csv_path
        self.val_csv_path = Path(val_csv_path) if val_csv_path else val_csv_path
        self.test_csv_path = Path(test_csv_path) if test_csv_path else test_csv_path
        self.tissue_mask_dir = Path(tissue_mask_dir) if tissue_mask_dir else tissue_mask_dir
        self.mask_suffix = mask_suffix

        self.level = level
        self.image_size = image_size
        self.tile_stride = tile_stride
        self.tile_size = tile_size
        self.network_output_stride = network_output_stride
        self.variable_input_shapes = variable_input_shapes
        self.num_workers = num_workers
        self.copy_to_gpu = copy_to_gpu
        self.transform = transform
        self.verbose = verbose
        self.filetype = filetype


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # assign a dataset
        if stage == "fit":
            self.train_dataset = StreamingClassificationDataset(
                self.image_dir,
                csv_file=self.train_csv_path,
                tile_size=self.tile_size,
                img_size=self.image_size,
                read_level=self.level,
                transform=self.transform,
                mask_dir=self.tissue_mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )
            self.sampler = weighted_sampler(self.train_dataset)

            self.val_dataset = StreamingClassificationDataset(
                self.image_dir,
                csv_file=self.val_csv_path,
                tile_size=self.tile_size,
                img_size=self.image_size,
                read_level=self.level,
                transform=None,
                mask_dir=self.tissue_mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )

        if stage == "test":
            self.test_dataset = StreamingClassificationDataset(
                self.image_dir,
                csv_file=self.test_csv_path,
                tile_size=self.tile_size,
                img_size=self.image_size,
                read_level=self.level,
                transform=None,
                mask_dir=self.tissue_mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )
        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            sampler=self.sampler,
            shuffle=False,
            prefetch_factor=1,
            pin_memory=False,
            batch_size=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=1,
            pin_memory=False,
            batch_size=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=1,
            pin_memory=False,
            batch_size=1,
        )

    def predict_dataloader(self):
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer image to gpu only if copy_to_gpu is True
        DDP bug?: for some reason when training with more than 1 gpu, the batches will still be transferred to gpu
        somewhere between this function and the forward step in the model, making this function useless

        batch : {image: image, mask: mask}, label, fname
        batch : {image: image}, label, fname
        """

        batch[2] = batch[2][0]
        # Always put mask to gpu
        if batch[0]["mask"] is not None:
            batch[0]["mask"] = batch[0]["mask"].to(device)

        if not self.copy_to_gpu:
            batch[0]["image"] = batch[0]["image"].to("cpu")
            batch[1] = batch[1].to(device)

            return batch
        batch[0]["image"] = batch[0]["image"].to(device)
        batch[1] = batch[1].to(device)

        return batch

