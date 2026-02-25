import lightning as L
import torch
import albumentationsxl as A

from pathlib import Path

from torch.utils.data import DataLoader

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from streamingclam.datatwo.streamingdataset import StreamingDataset
from streamingclam.datatwo.samplers import weighted_sampler
from streamingclam.datatwo.utils import streaming_collate_fn


class BatchWrapper:
    """ For some reason pytorch lightning auto transfers images to cuda, even if correct callbacks are used.
    This is troublesome for streaming if input images are large (>32k).
    The batch wrapper class is a non-standard format that will not transfer anythint o cuda without explicitly
    doing so.
    """
    def __init__(self, batch_dict: dict):
        for key, value in batch_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        keys = ', '.join(f"{k}={getattr(self, k).device if hasattr(getattr(self, k), 'device') else type(getattr(self, k)).__name__}"
                         for k in self.__dict__)
        return f"BatchWrapper({keys})"

class StreamingDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_dir: Path | str,
        read_level: int,
        tile_size: int,
        tile_stride: int,
        network_output_stride: int,
        train_csv_path: str | Path | None = None,
        val_csv_path: str | Path | None = None,
        test_csv_path: str | Path | None = None,
        attention_csv_path: str | Path | None = None,
        mask_dir: str | Path | None = None,
        mask_suffix: str | None = None,
        normalize: bool = True,
        image_size: int | None = None,
        variable_input_shapes: bool = True,
        copy_to_gpu: bool = False,
        num_workers: int = 2,
        transform: A.BaseCompose | None = None,
        verbose: bool = True,
        filetype: str = ".tif",
        distributed: bool = False,
    ):
        super().__init__()
        self.image_dir = image_dir

        # Only for training, during testing only the data dir is used
        self.train_csv_path = Path(train_csv_path) if train_csv_path else train_csv_path
        self.val_csv_path = Path(val_csv_path) if val_csv_path else val_csv_path
        self.test_csv_path = Path(test_csv_path) if test_csv_path else test_csv_path
        self.att_csv_path = Path(attention_csv_path) if attention_csv_path else None
        self.mask_dir = Path(mask_dir) if mask_dir else mask_dir
        self.mask_suffix = mask_suffix
        self.normalize = normalize

        self.level = read_level
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
        self.is_distributed = distributed

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # assign a dataset

        if stage == "fit":
            self.train_dataset = StreamingDataset(
                self.image_dir,
                csv_path=self.train_csv_path,
                tile_size=self.tile_size,
                image_size=self.image_size,
                read_level=self.level,
                transform=self.transform,
                normalize=self.normalize,
                mask_dir=self.mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )
            self.train_sampler = weighted_sampler(self.train_dataset, distributed=self.is_distributed)

            self.val_dataset = StreamingDataset(
                self.image_dir,
                csv_path=self.val_csv_path,
                tile_size=self.tile_size,
                image_size=self.image_size,
                read_level=self.level,
                transform=None,
                normalize=self.normalize,
                mask_dir=self.mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )
            self.val_sampler = torch.utils.data.DistributedSampler(self.val_dataset, shuffle=False) if self.is_distributed else None


        if stage == "test":
            self.test_dataset = StreamingDataset(
                self.image_dir,
                csv_path=self.test_csv_path,
                tile_size=self.tile_size,
                image_size=self.image_size,
                read_level=self.level,
                transform=None,
                normalize=self.normalize,
                mask_dir=self.mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
            )
        if stage == "predict":
            self.predict_dataset = StreamingDataset(
                self.image_dir,
                csv_path=self.test_csv_path,
                tile_size=self.tile_size,
                image_size=self.image_size,
                read_level=self.level,
                transform=None,
                normalize=self.normalize,
                mask_dir=self.mask_dir,
                mask_suffix=self.mask_suffix,
                variable_input_shapes=self.variable_input_shapes,
                tile_stride=self.tile_stride,
                network_output_stride=self.network_output_stride,
                filetype=self.filetype,
                heatmap_mode=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            shuffle=False,
            prefetch_factor=1,
            pin_memory=False,
            batch_size=1,
            collate_fn=streaming_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=None,
            pin_memory=False,
            batch_size=1,
            collate_fn=streaming_collate_fn,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=None,
            pin_memory=False,
            batch_size=1,
            collate_fn=streaming_collate_fn
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=None,
            pin_memory=False,
            batch_size=1,
            collate_fn=streaming_collate_fn
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer image to gpu only if copy_to_gpu is True
        DDP bug?: for some reason when training with more than 1 gpu, the batches will still be transferred to gpu
        somewhere between this function and the forward step in the model, making this function useless

        batch : {image: image, mask: mask}, covariates, label, fname, metadata
        batch : {image: image}, covariates, label, fname, metadata
        """

        # Always put mask to gpu
        if "mask" in batch.keys() and batch['mask'] is not None:
            batch["mask"] = batch["mask"].to(device)

        batch["image"] = batch["image"].to("cpu") if not self.copy_to_gpu else batch["image"].to(device)
        batch["label"] = batch["label"].to(device)

        return BatchWrapper(batch)