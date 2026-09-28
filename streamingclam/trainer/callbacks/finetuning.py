from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.rank_zero import rank_zero_info



class SafeStreamUnfreezing(BaseFinetuning):
    def __init__(self, unfreeze_epoch: int = 2):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.stream_network, train_bn=False)
        pl_module.train_streaming_layers = False
        rank_zero_info("🔒 stream_network frozen at start")

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == self.unfreeze_epoch:
            rank_zero_info(f"[Rank {pl_module.global_rank}] 🔓 Unfreezing stream_network at epoch {epoch}")

            # Unfreeze and set LR on predefined param groups
            BaseFinetuning.make_trainable(pl_module.stream_network)