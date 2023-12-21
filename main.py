import torch
import warnings

import pandas as pd
import lightning.pytorch as pl

from pprint import pprint
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint


from streamingclam.models.streamingclam import StreamingCLAM
from streamingclam.dataloaders.dataset import StreamingClassificationDataset
from streamingclam.dataloaders.sampler import weighted_sampler
from streamingclam.options import TrainConfig
from streamingclam.utils.memory_format import MemoryFormat


def prepare_dataset(csv_file, options):
    return StreamingClassificationDataset(
        img_dir=options.image_path,
        csv_file=csv_file,
        tile_size=options.tile_size,
        img_size=options.img_size,
        transform=[],
        mask_dir=options.mask_path,
        mask_suffix=options.mask_suffix,
        variable_input_shapes=options.variable_input_shapes,
        tile_delta=options.tile_delta,
        network_output_stride=options.network_output_stride,
        filetype=options.filetype,
        read_level=options.read_level,
    )


def prepare_dataloaders(options):
    train_dataset = prepare_dataset(options.train_csv, options)
    val_dataset = prepare_dataset(options.val_csv, options)
    test_dataset = prepare_dataset(options.test_csv, options)

    sampler = weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        num_workers=options.num_workers,
        sampler=sampler,
        shuffle=False,
        prefetch_factor=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, num_workers=options.num_workers, shuffle=False, prefetch_factor=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, num_workers=options.num_workers, shuffle=False, prefetch_factor=1, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def configure_callbacks(options):
    checkpoint_callback = ModelCheckpoint(
        dirpath=options.default_save_dir + "/checkpoints",
        monitor="val_loss",
        filename="streamingclam-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        verbose=True,
    )
    return checkpoint_callback


def configure_checkpoints():
    try:
        # Check for last checkpoint
        last_checkpoint = list(Path(options.default_save_dir + "/checkpoints").glob("*last.ckpt"))
        last_checkpoint_path = str(last_checkpoint[0])
    except IndexError:
        if options.resume:
            warnings.warn("Resume option enabled, but no checkpoint files found. Training will start from scratch.")
        last_checkpoint_path = None

    return last_checkpoint_path


def configure_trainer(options):
    checkpoint_callback = configure_callbacks(options)
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu",
        max_epochs=options.num_epochs,
        devices=options.num_gpus,
        strategy=options.strategy,
        callbacks=[checkpoint_callback, MemoryFormat()],
        accumulate_grad_batches=options.grad_batches,
        precision="16-true",
    )

    return trainer


def get_model_statistics(model):
    """Prints model statistics for reference purposes

    Prints network output strides, and tile delta for streaming

    Parameters
    ----------
    model : pytorch lightning model object


    """

    tile_delta = model.configure_tile_delta()
    network_output_stride = max(
        model.stream_network.output_stride[1] * options.max_pool_kernel, model.stream_network.output_stride[1]
    )

    print("")
    print("=============================")
    print(" Network statistics")
    print("=============================")

    print("Network output stride")

    print("tile delta", tile_delta)
    print("network output stride calc", model.stream_network.output_stride[1])
    print("Network max pool kernel", model.max_pool_kernel)
    print("Total network output stride", network_output_stride)

    print("Network tile delta", tile_delta)
    print("==============================")
    print("")

    return tile_delta, network_output_stride


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))

    pprint(options)

    if options.mode == "train":
        model = StreamingCLAM(
            options.encoder,
            tile_size=options.tile_size,
            loss_fn=torch.nn.CrossEntropyLoss(),
            branch=options.branch,
            n_classes=options.num_classes,
            max_pool_kernel=options.max_pool_kernel,
            statistics_on_cpu=options.statistics_on_cpu,
            verbose=options.verbose,
            train_streaming_layers=options.train_streaming_layers,
            dtype=torch.float16,
            normalize_on_gpu=True,
        )

        tile_delta, network_output_stride = get_model_statistics(model)

        options.tile_delta = tile_delta
        options.network_output_stride = network_output_stride

        train_loader, val_loader, test_loader = prepare_dataloaders(options)
        trainer = configure_trainer(options)

        # Train model
        # for gradient checkpointing: https: // lightning.ai / docs / pytorch / stable / advanced / training_tricks.html

        last_checkpoint_path = configure_checkpoints()
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=last_checkpoint_path if (options.resume and last_checkpoint_path) else None,
        )

    elif options.mode == "validation":
        model = StreamingCLAM.load_from_checkpoint(
            "/data/pathology/projects/pathology-bigpicture-streamingclam/lightstream-implementation/ckp/sclam_res34_sb_cam16_65k_0.5mpp.ckpt",
            encoder="resnet34",
            tile_size=options.tile_size,
            loss_fn=torch.nn.functional.cross_entropy,
            branch="sb",
            n_classes=2,
            copy_to_gpu=False,
            normalize_on_gpu=True,
            verbose=True,
        )

        tile_delta, network_output_stride = get_model_statistics(model)

        options.tile_delta = tile_delta
        options.network_output_stride = network_output_stride
        train_loader, val_loader, test_loader = prepare_dataloaders(options)
        trainer = configure_trainer(options)

        trainer.validate(model, dataloaders=val_loader)

    elif options.mode == "test":
        model = StreamingCLAM.load_from_checkpoint(
            "/data/pathology/projects/pathology-bigpicture-streamingclam/lightstream-implementation/ckp/sclam_res34_sb_cam16_65k_0.5mpp.ckpt",
            encoder="resnet34",
            tile_size=options.tile_size,
            loss_fn=torch.nn.functional.cross_entropy,
            branch="sb",
            n_classes=2,
            copy_to_gpu=False,
            normalize_on_gpu=True,
            verbose=True,
            max_pool_kernel=options.max_pool_kernel,
        )

        tile_delta, network_output_stride = get_model_statistics(model)

        options.tile_delta = tile_delta
        options.network_output_stride = network_output_stride

        train_loader, val_loader, test_loader = prepare_dataloaders(options)
        trainer = configure_trainer(options)

        trainer.test(
            model,
            dataloaders=test_loader,
        )

        all_outputs = model.test_outputs

        df = pd.DataFrame(all_outputs, columns=["slide_id", "Y", "Y_hat", "p_0", "p_1"])
        df.to_csv(options.default_save_dir + "/test_output.csv", index=False)

    else:
        raise NotImplementedError(f"mode must be one of (train, validation, test), but found {options.mode}")
