import torch
import warnings

import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pathlib import Path

from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

# My own edits here
from applications.streamingclam.streamingclam_regression import StreamingCLAM
from applications.data.dataset_regression import StreamingSurvivalDataset
from applications.options import TrainConfig

wandb_logger = WandbLogger(project="Bladder")


def weighted_sampler(dataset):
    labels = np.array([int(label) for label in dataset.labels])

    # more generalized approach, should result in the same distribution
    # calculate inverse class frequency, then squash to [0,1] by dividing by max value
    _, class_counts = np.unique(labels, return_counts=True)
    inv_freq = len(labels) / class_counts
    norm_weights = inv_freq / np.max(inv_freq)

    # create weight array and replace labels by their weights
    weights = np.array(labels, dtype=np.float32)
    for i, weight in enumerate(norm_weights):
        weights[labels == i] = weight

    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def prepare_dataset(csv_file, options):
    return StreamingSurvivalDataset(
        img_dir=options.image_path,
        csv_file=csv_file,
        tile_size=options.tile_size,
        img_size=options.img_size,
        transform=[],
        mask_dir=options.mask_path,
        mask_suffix=options.mask_suffix,
        variable_input_shapes=options.variable_input_shapes,
        tile_delta=tile_delta,
        network_output_stride=network_output_stride,
        filetype=options.filetype,
        read_level=options.read_level,
    )


if __name__ == "__main__":
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    print(args)
    options.parser_to_options(vars(args))

    print(options)

    model = StreamingCLAM(
        options.encoder,
        tile_size=options.tile_size,
        loss_fn=torch.nn.SmoothL1Loss(),
        branch=options.branch,
        n_classes=options.num_classes,
        max_pool_kernel=options.max_pool_kernel,
        statistics_on_cpu=options.statistics_on_cpu,
        verbose=options.verbose,
        train_streaming_layers=options.train_streaming_layers,
        dtype=torch.float16,
    )

    tile_delta = model._configure_tile_delta()
    network_output_stride = max(
        model.stream_network.output_stride[1] * model.max_pool_kernel, model.stream_network.output_stride[1]
    )
    print("tile delta", tile_delta)
    print("network output stride calc", network_output_stride)

    train_dataset = prepare_dataset(options.train_csv, options)
    val_dataset = prepare_dataset(options.val_csv, options)
    test_dataset = prepare_dataset(options.test_csv, options)

    sampler = weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, num_workers=options.num_workers, sampler=sampler, shuffle=False, prefetch_factor=1
    )
    val_loader = DataLoader(val_dataset, num_workers=options.num_workers, shuffle=False, prefetch_factor=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath=options.default_save_dir + "/checkpoints",
        monitor="val_loss",
        filename="streamingclam-derma-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        verbose=True,
    )

    try:
        # Check for last checkpoint
        last_checkpoint = list(Path(options.default_save_dir + "/checkpoints").glob("*last.ckpt"))
        last_checkpoint_path = str(last_checkpoint[0])
    except IndexError:
        if options.resume:
            warnings.warn("Resume option enabled, but no checkpoint files found. Training will start from scratch.")
        last_checkpoint_path = None

    # Train model
    # for gradient checkpointing: https: // lightning.ai / docs / pytorch / stable / advanced / training_tricks.html
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu",
        max_epochs=options.num_epochs,
        devices=options.num_gpus,
        strategy=options.strategy,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=options.grad_batches,
        precision="16-mixed",
        logger=wandb_logger,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=last_checkpoint_path if (options.resume and last_checkpoint_path) else None,
    )
