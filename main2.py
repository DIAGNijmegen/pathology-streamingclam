import os

os.environ["WANDB_DIR"] = "/home/stephandooper"
os.environ["VIPS_CONCURRENCY"] = "30"
os.environ["OMP_NUM_THREADS"] = "4"
import pyvips

pyvips.cache_set_max(20)
pyvips.cache_set_max_mem(1024 * 1024)

import torch
import warnings

from pathlib import Path
from pprint import pprint

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from streamingclam.options import TrainConfig
from streamingclam.utils.memory_format import MemoryFormat
from streamingclam.utils.printing import PrintingCallback
from streamingclam.data.splits import StreamingCLAMDataModule
from streamingclam.data.dataset import augmentations
from streamingclam.models.sclam import StreamingCLAM

torch.set_float32_matmul_precision("medium")


def configure_callbacks(options):
    filename = "streamingclam-head-{epoch:02d}-{val_loss:.2f}-{valid_acc:.2f}-{valid_auc:.2f}"
    if options.train_streaming_layers:
        filename="streamingclam-all-{epoch:02d}-{val_loss:.2f}-{valid_acc:.2f}-{valid_auc:.2f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=options.default_save_dir + "/checkpoints",
        monitor="val_loss",
        filename=filename,
        save_top_k=3,
        save_last=True,
        mode="min",
        verbose=True,
    )
    return checkpoint_callback


def configure_checkpoints():
    try:
        # Check for last checkpoint
        last_checkpoint = list(
            Path(options.default_save_dir + "/checkpoints").glob("*last.ckpt")
        )
        last_checkpoint_path = str(last_checkpoint[0])

        print(f"Checkpoint path found at {last_checkpoint_path}")
    except IndexError:
        if options.resume:
            warnings.warn(
                "Resume option enabled, but no checkpoint files found. Training will start from scratch."
            )
        last_checkpoint_path = None

    return last_checkpoint_path


def get_model_statistics(model):
    """Prints model statistics for reference purposes

    Prints network output strides, and tile delta for streaming

    Parameters
    ----------
    model : pytorch lightning model object


    """

    tile_stride = model.configure_tile_stride()
    network_output_stride = model.stream_network.output_stride[1]

    return tile_stride, network_output_stride


def configure_trainer(options, finetune=False):
    checkpoint_cb = configure_callbacks(options)
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu",
        max_epochs=options.num_epochs_head
        if not finetune
        else options.num_epochs_finetune,
        devices=options.num_gpus,
        accumulate_grad_batches=options.grad_batches,
        precision=options.precision,
        callbacks=[checkpoint_cb, MemoryFormat(), PrintingCallback(options)],
        strategy=options.strategy,
        benchmark=False,
        logger=wandb_logger,
        gradient_clip_val=0.5 if finetune else 0,
        gradient_clip_algorithm="norm",
    )
    return trainer


def get_streaming_options(options):
    fields = [
        "statistics_on_cpu",
        "normalize_on_gpu",
        "copy_to_gpu",
        "verbose",
    ]
    opt_dict = options.to_dict()
    return {key: opt_dict[key] for key in fields}


def configure_streamingclam(options, streaming_options, finetune=False):
    sclam_opts = {
        "encoder": options.encoder,
        "tile_size": options.tile_size,
        "loss_fn": options.loss_fn,
        "branch": options.branch,
        "n_classes": options.num_classes,
        "max_pool_kernel": options.max_pool_kernel,
        "stream_max_pool_kernel": options.stream_max_pool_kernel,
        "train_streaming_layers": options.train_streaming_layers,
        "instance_eval": options.instance_eval,
        "return_features": options.return_features,
        "attention_only": options.attention_only,
        "learning_rate": options.learning_rate,
    }

    if options.mode == "fit":
        if finetune:
            print("Loading head checkpoint to start training all layers")
            sclam_opts["tile_size"] = options.tile_size_finetune
            sclam_opts["train_streaming_layers"] = True
            model = StreamingCLAM.load_from_checkpoint(options.last_checkpoint_path, **sclam_opts, **streaming_options)
        else:
            model = StreamingCLAM(
                **sclam_opts,
                **streaming_options,
            )
    else:
        model = StreamingCLAM.load_from_checkpoint(
            options.ckp_file,
            **sclam_opts,
            **streaming_options,
        )
    return model


def configure_datamodule(options):
    dm = StreamingCLAMDataModule(
        image_dir=options.image_path,
        level=options.read_level,
        tile_size=options.tile_size,
        tile_stride=options.tile_stride,
        network_output_stride=options.network_output_stride,
        train_csv_path=options.train_csv,
        val_csv_path=options.val_csv,
        test_csv_path=options.test_csv,
        tissue_mask_dir=options.mask_path,
        mask_suffix=options.mask_suffix,
        image_size=options.image_size,
        variable_input_shapes=options.variable_input_shapes,
        copy_to_gpu=options.copy_to_gpu,
        num_workers=options.num_workers,
        transform=augmentations
        if (options.use_augmentations and options.mode == "fit")
        else None,
    )

    return dm


def prepare_trainer(options, finetune=False):
    model = configure_streamingclam(options, streaming_options, finetune=finetune)

    tile_stride, network_output_stride = get_model_statistics(model)

    options.tile_stride = tile_stride
    options.network_output_stride = max(
        network_output_stride * options.max_pool_kernel, network_output_stride
    )

    dm = configure_datamodule(options)
    dm.setup(stage=options.mode)
    trainer = configure_trainer(options, finetune=finetune)

    return trainer, model, dm


if __name__ == "__main__":
    pl.seed_everything(1)
    wandb_logger = WandbLogger(
        project="lightstreamingclam-test", save_dir="/home/stephandooper"
    )
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))
    # pprint(options)
    streaming_options = get_streaming_options(options)

    if options.mode == "fit":
        trainer, model, dm = prepare_trainer(options)

        # log gradients, parameter histogram and model topology
        if trainer.global_rank == 0:
            print("at rank 0, logging wandb config")
            wandb_logger.experiment.config.update(options.to_dict())

        version = wandb_logger.version

        last_checkpoint_path = configure_checkpoints()
        # model.head = torch.compile(model.head)
        # model.stream_network.stream_module = torch.compile(model.stream_network.stream_module)

        print("Starting training")
        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=last_checkpoint_path
            if (options.resume and last_checkpoint_path)
            else None,
        )
        print("finished training head, preparing finetuning stage")

        wandb_logger = WandbLogger(
            project="lightstreamingclam-test",
            save_dir="/home/stephandooper",
            version=version,
            resume="must",
        )

        last_checkpoint_path = configure_checkpoints()
        options.last_checkpoint_path = last_checkpoint_path
        trainer, model, dm = prepare_trainer(options, finetune=True)

        # Fine Tune
        trainer.fit(
            model,
            datamodule=dm,
        )

    elif options.mode == "test":
        checkpoint_path = configure_checkpoints()
        trainer.test(
            model=model,
            datamodule=dm,
        )

    elif options.mode == "predict":
        print("not implemented")
    else:
        raise ValueError(
            "mode must be one of fit, test or predict, found {}".format(options.mode)
        )

# DO:
