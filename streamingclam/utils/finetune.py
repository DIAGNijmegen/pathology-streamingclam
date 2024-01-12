import torch
from lightning.pytorch.callbacks import BaseFinetuning
from typing import Callable
import logging
log = logging.getLogger(__name__)

def multiplicative(epoch: int) -> float:
    return 2.0


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch: int=40,
                 lambda_func: Callable = multiplicative,
                 backbone_initial_lr: float=2e-8,
                 backbone_initial_ratio_lr: float = 1e-2,
                 rounding: int = 12,
                 should_align: bool = True,
                 verbose: bool = True):
        super().__init__()
        print("unfreezing streaming network at epoch", unfreeze_at_epoch)
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.lambda_func = lambda_func
        self.switch = True
        self.backbone_initial_lr = backbone_initial_lr
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.rounding: int = rounding
        self.verbose = verbose
        self.should_align = should_align

    def freeze_before_training(self, pl_module):
        # freeze any module you want

        # Here, we are freezing `feature_extractor`
        pl_module.train_streaming_layers = False
        self.freeze(pl_module.stream_network.stream_module, train_bn=False)

        # Just for good measure.
        pl_module.freeze_streaming_normalization_layers()

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is self._unfreeze_at_epoch, feature_extractor will start training.
        # Check this for every epoch in case we are resuming after failure


        if current_epoch == self._unfreeze_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr

            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        if current_epoch >= self._unfreeze_at_epoch:
            if self.switch:
                pl_module.train_streaming_layers = True
                self.unfreeze_and_add_param_group(
                    modules=pl_module.stream_network.stream_module,
                    optimizer=optimizer,
                    train_bn=False,
                    lr=initial_backbone_lr
                )

                print("Switching to training all layers in the network")
                self.switch = False

        if current_epoch > self._unfreeze_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(current_epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)},"
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )


    def on_train_epoch_end(self, trainer, pl_module):
        # Adjust streaming for the new situation, and unfreeze in the next epoch
        # Also change the dataloader with new tile size and tile stride.

        if trainer.current_epoch == (self._unfreeze_at_epoch - 1):
            print("preparing to finetune all layers for next epoch")
            print(
                "adjusting streaming model and dataloaders to new tile size and tile stride"
            )
            pl_module.to(memory_format=torch.contiguous_format)
            pl_module.tile_size = 7680
            pl_module.tile_cache_fname = None
            pl_module.disable_streaming_hooks()

            old_stream_network_dtype = pl_module.stream_network.dtype
            old_stream_module_dtype = next(pl_module.stream_network.stream_module.parameters()).dtype


            tile_cache = pl_module.load_tile_cache_if_needed()

            # Reset tile cache
            pl_module.constructor.tile_size = 7680
            pl_module.constructor.tile_cache = tile_cache
            pl_module.constructor.verbose = False
            pl_module.constructor.model.to(memory_format=torch.contiguous_format)
            pl_module.constructor.model.to(torch.float32)
            pl_module.stream_network = pl_module.constructor.prepare_streaming_model()
            pl_module.save_tile_cache_if_needed()

            # Put back dtype and memoryformat
            pl_module.stream_network.dtype = old_stream_network_dtype
            pl_module.stream_network.stream_module.to(old_stream_module_dtype)
            pl_module.stream_network.stream_module.to(memory_format=torch.channels_last)
            pl_module.on_train_start()
            # Reset dataloaders
            tile_stride = pl_module.configure_tile_stride()
            trainer.datamodule.tile_size = 7680
            trainer.datamodule.tile_stride = tile_stride
            trainer.datamodule.verbose = False
            trainer.datamodule.setup("fit")
