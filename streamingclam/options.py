import dataclasses

import torch.nn
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
    mode: str = "fit"  # fit, validation, test, featurize or predict
    unfreeze_streaming_layers_at_epoch: int = 20

    # Trainer options
    num_epochs: int = 50  # The number of epochs to train (max)
    strategy: str = "ddp_find_unused_parameters_true"
    default_save_dir: str = "/data/pathology/projects/pathology-bigpicture-streamingclam/lightstream-implementation/ckp"
    ckp_path: str = ""  # the name fo the ckp file within the default_save_dir
    resume: bool = True  # Whether to resume training from the last/best epoch
    grad_batches: int = 2  # Gradient accumulation: the amount of batches before optimizer step
    num_gpus: int = 4
    precision: str = "16-mixed"

    # StreamingClam options
    encoder: str = "resnet34"  # Resnet 18, ResNet34, Resnet50
    branch: str = "sb"  # sb or mb
    max_pool_kernel: int = 8
    num_classes: int = 2
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()
    instance_eval: bool = False
    return_features: bool = False
    attention_only: bool = False
    stream_max_pool_kernel: bool = False
    learning_rate: float = 2e-4  # the learning rate when training the CLAM head,
                                 # the finetuning callback defined in finetuned.py will handle the optimizer for all layers

    # Streaming options
    tile_size: int = 9984
    statistics_on_cpu: bool = True
    verbose: bool = True
    train_streaming_layers: bool = False
    normalize_on_gpu: bool = True
    copy_to_gpu: bool = False  # Whether to copy the entire image to the gpu. Recommended False if image > 16000x16000

    # Dataloader options
    image_size: int = 65536  # represents image size if variable_input_shape=False, else the maximum image size
    variable_input_shapes: bool = True
    filetype: str = ".tif"
    read_level: int = 1  # the level of the tif file (0 is highest resolution)
    num_workers: int = 3
    use_augmentations: bool = True

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
