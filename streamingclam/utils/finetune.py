import torch
from lightning.pytorch.callbacks import BaseFinetuning


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=40):
        super().__init__()
        print("unfreezing streaming network at epoch", unfreeze_at_epoch)
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.switch = True

    def freeze_before_training(self, pl_module):
        # freeze any module you want

        # Here, we are freezing `feature_extractor`
        pl_module.train_streaming_layers = False
        self.freeze(pl_module.stream_network.stream_module, train_bn=False)

        # Just for good measure and setting it to eval.
        pl_module.freeze_streaming_normalization_layers()

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is self._unfreeze_at_epoch, feature_extractor will start training.

        if current_epoch >= self._unfreeze_at_epoch:
            if self.switch:
                pl_module.train_streaming_layers = True
                self.unfreeze_and_add_param_group(
                    modules=pl_module.stream_network.stream_module,
                    optimizer=optimizer,
                    train_bn=False,
                )

                print("Switching to training all layers in the network")
                self.switch = False

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
