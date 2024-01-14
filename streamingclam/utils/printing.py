from lightning.pytorch.callbacks import Callback
from pprint import pprint

class PrintingCallback(Callback):
    def __init__(self, options):
        super().__init__()
        self.options = options

    def setup(self, trainer, pl_module, stage):
        pl_module.print(self.options)
        pl_module.print(self.options.to_dict())
        if trainer.global_rank == 0:
            print("Using configuration with the following options")
            pprint(self.options)

            print("")
            print("=============================")
            print(" Network statistics")
            print("=============================")

            print("Network output stride")

            print("tile delta", self.options.tile_stride)
            print("network output stride calc", pl_module.stream_network.output_stride[1])
            print("Total network output stride", self.options.network_output_stride)

            print("Network tile delta", self.options.tile_stride)
            print("==============================")
            print("")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
