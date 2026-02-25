import warnings
import torch
import os

torch.set_float32_matmul_precision("high")

from typing import Optional
from pathlib import Path

import lightning.pytorch as pl

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from streamingclam.datatwo.streamingdatamodule import StreamingDataModule
from streamingclam.trainer.sclam import StreamingCLAM
from streamingclam.trainer.options import StreamingCLAMOptions
from streamingclam.trainer.callbacks.printing import PrintingCallback
from streamingclam.trainer.callbacks.streamingwriter import StreamingHeatmapWriter
from streamingclam.trainer.callbacks.ddpsync import OptimizerStateSyncCheck


# TODO: graceful resuming: after training head, optimizer states are not restored, leading to errors, or fresh restarts
# TODO: cleanup the state loading routines, it's a mess.


def save_wandb_id(run_id: str, dest: Path | str) -> None:
    """
    Save the WandB run ID to a plain text file.
    """
    dest = Path(dest) / Path("wandb_id.txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(run_id.strip())


def load_wandb_id(path: str | Path) -> Optional[str]:
    """
    Load the WandB run ID from a plain text file.
    Returns None if the file doesn't exist or is empty.
    """
    path = Path(path) / Path("wandb_id.txt")
    if not path.exists():
        return None
    run_id = path.read_text().strip()
    return run_id if run_id else None


@rank_zero_only
def setup_wandb_logger(options: StreamingCLAMOptions) -> WandbLogger:
    wandb_run_id = load_wandb_id(options.trainer_options.default_root_dir)

    os.environ["WANDB_API_KEY"] = options.misc_options.wandb_api_key

    wandb_logger = WandbLogger(
        project=options.misc_options.wandb_project_name,
        name=options.trainer_options.default_root_dir.rsplit("/")[-1],
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

    print("Wandb run id: ", wandb_logger.experiment.id)
    save_wandb_id(wandb_logger.experiment.id, options.trainer_options.default_root_dir)
    return wandb_logger


def configure_trainer(options: StreamingCLAMOptions) -> pl.Trainer:
    logger = None
    if options.misc_options.use_wandb:
        wandb_logger = setup_wandb_logger(options)
        logger = wandb_logger

    loss_ckp = ModelCheckpoint(
        dirpath=options.trainer_options.default_root_dir + "/checkpoints",
        monitor="val/ll_loss",
        filename="sclam-loss-{epoch:02d}-val_total_loss={val/total_loss:.2f}-val_auc={val/auc:.2f}-val_acc={val/accuracy:.2f}",
        save_top_k=1,
        save_last=True,
        mode="max",
        verbose=True,
        auto_insert_metric_name=False,
    )

    auc_ckp = ModelCheckpoint(
        dirpath=options.trainer_options.default_root_dir + "/checkpoints",
        monitor="val/accuracy",
        filename="sclam-acc-{epoch:02d}-val_total_loss={val/total_loss:.2f}-val_auc={val/auc:.2f}-val_acc={val/accuracy:.2f}",
        save_top_k=1,
        save_last=False,
        mode="max",
        verbose=True,
        auto_insert_metric_name=False,
    )

    acc_ckp = ModelCheckpoint(
        dirpath=options.trainer_options.default_root_dir + "/checkpoints",
        monitor="val/auc",
        filename="sclam-auc-{epoch:02d}-val_total_loss={val/total_loss:.2f}-val_auc={val/auc:.2f}-val_acc={val/accuracy:.2f}",
        save_top_k=1,
        save_last=False,
        mode="max",
        verbose=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(monitor="val/ll_loss", min_delta=0.00, patience=100, verbose=True, mode="max")

    trainer = pl.Trainer(
        **options.trainer_options.to_dict(),
        callbacks=[
            loss_ckp,
            acc_ckp,
            auc_ckp,
            early_stopping,
            PrintingCallback(options.to_flat_dict()),
            StreamingHeatmapWriter(
                options.trainer_options.default_root_dir,
                "batch",
                options.misc_options.write_level,
            ),
            ModelSummary(max_depth=2),
            OptimizerStateSyncCheck(),
        ],
        enable_progress_bar=True,
        use_distributed_sampler=False,
        logger=logger
    )
    # logger=wandb_logger should be in pl.Trainer function
    return trainer


def add_strides_to_options(model, options):
    tile_stride = model.configure_tile_stride()
    output_stride = model.stream_network.output_stride[1]

    if options.streaming_options.stream_pooling_kernel:
        output_stride = output_stride  # output stride is already multiplied with pooling layer
    else:
        output_stride = max(output_stride * options.streaming_options.pooling_kernel, output_stride)

    # Add output stride and tile stride to the dataloader options, it's needed to resize masks correctly
    # and crop/pad images accordingly for streaming with variable input shapes.
    options.dataloader_options.tile_stride = int(tile_stride)
    options.dataloader_options.network_output_stride = int(output_stride)
    return options


def configure_model(options):
    model_options = options.model_options.to_dict()
    streaming_options = options.streaming_options.to_dict()
    return StreamingCLAM(**model_options, **streaming_options)


def configure_checkpoints_training(options):
    is_last = False
    try:
        # Check for last checkpoint
        checkpoint = list(Path(options.trainer_options.default_root_dir + "/checkpoints").glob("*last.ckpt"))
        checkpoint_path = str(checkpoint[0])
        is_last = True
    except IndexError:
        checkpoint_path = None
        if options.misc_options.resume:
            if options.misc_options.ckp_path == "":
                warnings.warn("Resume option enabled, but no checkpoint files found. Training from scratch")

            if options.misc_options.ckp_path != "":
                warnings.warn("Resume option enabled, but only custom ckp_path was found. Resuming from this stage")
                print(f"loading clam head at {options.misc_options.ckp_path}")
                checkpoint_path = torch.load(options.misc_options.ckp_path, weights_only=True)

    return checkpoint_path, is_last


def configure_checkpoints_inference(options: StreamingCLAMOptions) -> str:
    if not options.misc_options.ckp_path in ("", "last", "loss", "auc", "acc"):  # Assume it's a custom string
        assert Path(options.misc_options.ckp_path).exists()
        return options.misc_options.ckp_path

    if options.misc_options.ckp_path == "last":
        return str(list(Path(options.trainer_options.default_root_dir + "/checkpoints").glob("*last.ckpt"))[0])

    if options.misc_options.ckp_path == "loss":
        return str(list(Path(options.trainer_options.default_root_dir + "/checkpoints").glob("sclam-loss*.ckpt"))[0])

    if options.misc_options.ckp_path == "auc":
        return str(list(Path(options.trainer_options.default_root_dir + "/checkpoints").glob("sclam-auc*.ckpt"))[0])

    if options.misc_options.ckp_path == "acc":
        return str(list(Path(options.trainer_options.default_root_dir + "/checkpoints").glob("sclam-acc*.ckpt"))[0])


if __name__ == "__main__":
    pl.seed_everything(5)
    options = StreamingCLAMOptions.from_args()

    model = configure_model(options)
    options = add_strides_to_options(model, options)

    trainer = configure_trainer(options)
    distributed = trainer.world_size > 1
    datamodule = StreamingDataModule(
        tile_size=options.streaming_options.tile_size,
        **options.dataloader_options.to_dict(),
        distributed=distributed,
        normalize=not options.streaming_options.normalize_on_gpu,
    )
    print("Selected strategy:", type(trainer.strategy).__name__)

    if options.misc_options.stage == "fit":
        checkpoint_path, is_last = configure_checkpoints_training(options)
        if checkpoint_path is not None:
            if is_last:
                print("Resuming from last checkpoint", checkpoint_path)
                trainer.fit_loop.max_epochs = 100
                trainer.fit(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=checkpoint_path if options.misc_options.resume else None,
                )

            else:
                print("Loading from state dict from", options.misc_options.ckp_path)
                model_options = options.model_options.to_dict()
                streaming_options = options.streaming_options.to_dict()
                model = StreamingCLAM.load_from_checkpoint(
                    options.misc_options.ckp_path, **model_options, **streaming_options, strict=False
                )

                trainer = configure_trainer(options)

                trainer.fit(model=model, datamodule=datamodule)
        else:
            print("no pre-trained weights found, starting from scratch")
            trainer.fit(model=model, datamodule=datamodule)

    elif options.misc_options.stage == "test":
        ckp_path = configure_checkpoints_inference(options)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckp_path)
    elif options.misc_options.stage == "predict":
        if options.misc_options.ckp_path in ("last", "loss", "auc", "acc"):
            ckp_path = configure_checkpoints_inference(options)
            trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckp_path)
        else:
            model_options = options.model_options.to_dict()
            streaming_options = options.streaming_options.to_dict()
            model = StreamingCLAM.load_from_checkpoint(
                options.misc_options.ckp_path, **model_options, **streaming_options, strict=False
            )

            trainer = configure_trainer(options)

            trainer.predict(model=model, datamodule=datamodule)

    else:
        raise ValueError("mode must be one of fit, test or predict, found {}".format(options.mode))
