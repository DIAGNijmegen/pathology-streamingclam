import dataclasses
from dataclasses_json import dataclass_json
import argparse


@dataclass_json
@dataclasses.dataclass
class TrainConfig:
    image_path: str = ""
    mask_path: str = ""
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    mask_suffix: str = "_tissue"  # the suffix for mask tissues e.g. tumor_069_<mask_suffix>.tif
    default_save_dir: str = "/opt/ml/checkpoints"
    num_gpus: int = 8
    strategy: str = "ddp_find_unused_parameters_true"
    grad_batches: int = 8  # Gradient accumulation: the amount of batches before optimzier step
    resume: bool = True  # Whether to resume training from the last epochs

    # StreamingClam options
    num_epochs: int = 100  # The number of epochs to train (max)
    encoder: str = "resnet34"  # Resnet 18, ResNet34, Resnet50
    branch: str = "sb"  # sb or mb
    max_pool_kernel: int = 0
    num_classes: int = 2

    # Streaming options
    tile_size: int = 3200
    statistics_on_cpu: bool = True
    verbose: bool = False
    train_streaming_layers: bool = False

    # Dataloader options
    img_size: int = 32768  # represents image size if variable_input_shape=False, else the maximum image size
    variable_input_shapes: bool = True
    filetype: str = ".tif"
    read_level: int = 3  # the level of the tif file (0 is highest resolution)
    num_workers: int = 3

    def configure_parser_with_options(self):
        """Create an argparser based on the attributes"""
        parser = argparse.ArgumentParser(description="MultiGPU streaming")
        for name, default in dataclasses.asdict(self).items():
            argname = "--" + name
            tp = type(default)
            if tp is bool:
                if default == True:
                    argname = "--no_" + name
                    parser.add_argument(argname, action="store_false", dest=name)
                else:
                    parser.add_argument(argname, action="store_true")
            else:
                parser.add_argument(argname, default=default, type=tp)
        return parser

    def parser_to_options(self, parsed_args: dict):
        """Parse an argparser"""
        for name, value in parsed_args.items():
            self.__setattr__(name, value)
