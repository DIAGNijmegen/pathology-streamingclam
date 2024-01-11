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
from torch.utils.data import DataLoader


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy


from streamingclam.options import TrainConfig
from streamingclam.utils.memory_format import MemoryFormat
from streamingclam.utils.finetune import FeatureExtractorFreezeUnfreeze
from streamingclam.data.splits import StreamingCLAMDataModule
from streamingclam.data.dataset import augmentations
from streamingclam.data.feature_dataset import FeatureDataset
from streamingclam.models.sclam import StreamingCLAM
from streamingclam.utils.feature_writer import FeatureWriter

torch.set_float32_matmul_precision("medium")


class PrintingCallback(Callback):
    def __init__(self, options):
        super().__init__()
        self.options = options

    def setup(self, trainer, pl_module, stage):
        print("Using configuration with the following options")
        pl_module.print(self.options.to_dict())

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


def configure_callbacks(options):
    checkpoint_callback = ModelCheckpoint(
        dirpath=options.default_save_dir + "/checkpoints",
        monitor="val_loss",
        filename="streamingclam-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        verbose=True,
    )

    save_dir = str(Path(options.default_save_dir) / Path(options.encoder + "_features"))
    feature_cb = FeatureWriter(output_dir=str(save_dir), write_interval="batch")

    finetune_cb = FeatureExtractorFreezeUnfreeze(options.unfreeze_streaming_layers_at_epoch)
    return checkpoint_callback, finetune_cb, feature_cb


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
    checkpoint_callback, finetune_cb, feature_cb = configure_callbacks(options)
    trainer = pl.Trainer(
        default_root_dir=options.default_save_dir,
        accelerator="gpu",
        max_epochs=options.num_epochs,
        devices=options.num_gpus,
        accumulate_grad_batches=options.grad_batches,
        precision=options.precision,
        callbacks=[checkpoint_callback, MemoryFormat(), finetune_cb, PrintingCallback(options), feature_cb],
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True),
        benchmark=False,
        reload_dataloaders_every_n_epochs=options.unfreeze_streaming_layers_at_epoch,
        num_sanity_val_steps=0,
    )
    return trainer


def get_model_statistics(model, options):
    """Prints model statistics for reference purposes

    Prints network output strides, and tile delta for streaming

    Parameters
    ----------
    model : pytorch lightning model object


    """

    tile_stride = model.configure_tile_stride()
    network_output_stride = model.stream_network.output_stride[1]

    print("")
    print("=============================")
    print(" Network statistics")
    print("=============================")

    print("Network output stride")

    print("tile delta", tile_stride)
    print("network output stride calc", model.stream_network.output_stride[1])
    print("Total network output stride", network_output_stride)

    print("Network tile delta", tile_stride)
    print("==============================")
    print("")

    return tile_stride, network_output_stride


def get_streaming_options(options):
    fields = [
        "statistics_on_cpu",
        "normalize_on_gpu",
        "copy_to_gpu",
        "verbose",
    ]
    opt_dict = options.to_dict()
    return {key: opt_dict[key] for key in fields}


def configure_streamingclam(options, streaming_options):
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
        "return_resnet_features": options.mode == "featurize",
        "attention_only": options.attention_only,
    }

    if options.mode in ("fit", "featurize"):
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


if __name__ == "__main__":
    pl.seed_everything(1)
    wandb_logger = WandbLogger(project="lightstreamingclam-test", save_dir="/home/stephandooper")
    # Read json config from file
    options = TrainConfig()
    parser = options.configure_parser_with_options()
    args = parser.parse_args()
    options.parser_to_options(vars(args))
    #pprint(options)
    streaming_options = get_streaming_options(options)

    model = configure_streamingclam(options, streaming_options)
    tile_stride, network_output_stride = get_model_statistics(model, options)

    options.tile_stride = tile_stride
    options.network_output_stride = network_output_stride * options.max_pool_kernel

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
        transform=augmentations if (options.use_augmentations and options.mode == "fit") else None,
    )

    dm.setup(stage=options.mode)
    trainer = configure_trainer(options)

    if options.mode == "fit":
        # log gradients, parameter histogram and model topology
        if trainer.global_rank == 0:
            wandb_logger.experiment.config.update(options.to_dict())

        print("trainer strategy", trainer.strategy)
        last_checkpoint_path = configure_checkpoints()
        # model.head = torch.compile(model.head)
        # model.stream_network.stream_module = torch.compile(model.stream_network.stream_module)
        # print(model.stream_network)
        print("Starting training")

        trainer.fit(
            model=model,
            datamodule=dm,
            ckpt_path=last_checkpoint_path if (options.resume and last_checkpoint_path) else None,
        )


    elif options.mode == "test":
        checkpoint_path = configure_checkpoints()
        trainer.test(
            model=model,
            datamodule=dm,
        )

    elif options.mode == "predict":
        print("not implemented")

    elif options.mode == "featurize":
        """
        This mode will take all the images in the image directory specified in image_dir and optionally mask_dir
        Then, images will be parsed through the ResNet network and max pooling blocks. The resulting feature maps
        are then saved as <image_name>.pt  and <image_name_mask>.pt files in default_root_dir/features_<encoder>
        """
        feature_dataset = FeatureDataset(
            img_dir=options.image_path,
            tile_size=options.tile_size,
            img_size=options.image_size,
            mask_dir=options.mask_path,
            mask_suffix=options.mask_suffix,
            variable_input_shapes=options.variable_input_shapes,
            tile_stride=options.tile_stride,
            filetype=options.filetype,
            read_level=options.read_level,
        )

        feature_dataloader = DataLoader(
            feature_dataset,
            num_workers=options.num_workers,
            shuffle=False,
            prefetch_factor=1,
            pin_memory=False,
            batch_size=1,
        )

        # Will return imagenet features by default
        trainer.predict(
            model=model,
            dataloaders=feature_dataloader,
            ckpt_path=None #last_checkpoint_path if (options.resume and last_checkpoint_path) else None,
        )

    

    else:
        raise ValueError("mode must be one of fit, test or predict, found {}".format(options.mode))



