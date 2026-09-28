from lightning.pytorch.callbacks import Callback
from pprint import pprint

class PrintingCallback(Callback):
    def __init__(self, options):
        super().__init__()
        self.options = options

    def setup(self, trainer, pl_module, stage):
        pl_module.print(self.options)
        if trainer.global_rank == 0:
            print("Using configuration with the following options")
            pprint(self.options)

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")