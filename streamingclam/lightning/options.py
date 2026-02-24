import argparse
import torch
import types
from dataclasses import dataclass, fields, field
from typing import Dict, Any, Union, Optional
from dataclasses_json import dataclass_json, config
from streamingclam.datatwo.augmentations import augmentations


def encode_augmentations(aug):
    if aug:
        return augmentations
    return None

@dataclass_json
@dataclass
class StreamingCLAMOptions:
    @dataclass_json
    @dataclass
    class ModelOptions:
        num_classes: int
        encoder: str = "resnet34"  # Resnet 18, resNet34, or resnet39
        branch: str = "sb" # single branch (sb) or multi branch (mb) clam attention model
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss() # the primary MIL loss function
        instance_eval: bool = False # use the instance clustering in the CLAM module (default false)
        instance_loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss() # the instance clustering loss
        return_features: bool = False # Add M (in clam) to the results dict and return it (default false)
        attention_only: bool = False  # Return only the unnormalized attention (default false)
        stream_pooling_kernel: bool = False
        initial_lr: float = 1e-3  # the learning rate when training the CLAM attention head
        finetune_lr: float = 1e-3 # Learning rate for head + backbone after unfreezing (training full streaming model)
        unfreeze_epoch: int = 15 # Epoch to train all streaming layers
        accumulate_grad_batches: int = 4  # Gradient accumulation: the amount of batches before optimizer step
                                           # We use automatic_optimization=False, so it's not exposed to the trainer!

    @dataclass_json
    @dataclass
    class StreamingOptions:
        pooling_layer: str = "avgpool"  # one of maxpool, avgpool, none
        pooling_kernel: int = 16  # Kernel size & stride for the maxpool/avgpool
        stream_pooling_kernel: bool = False # Will add the pooling layer to the streaming network, otherwise done after
        tile_size: int = 3200  # The tile size on the gpu, as high as the gpu vram can handle (will not affect classification performance, only speed)
        statistics_on_cpu: bool = True  # Recommended to set to true since it takes up a lot of memory (only once)
        verbose: bool = True  # Verbose behaviour of streaming scnn.py
        normalize_on_gpu: bool = True  # Whether to normalize tiles on the GPU with ImageNet statistics. Default is True
        copy_to_gpu: bool = True  # Whether to copy the entire image to the gpu. Otherwise, only tile_size is copied
        tile_cache_path: str = ''

    @dataclass_json
    @dataclass
    class TrainerOptions:
        max_epochs: int = 100  # The number of epochs to train (max)
        default_root_dir: str = "" # full path to where all the results will be stored (weights, checkpoints, logs, etc)
        devices: int = 1 # The number of gpu's used for training
        precision: str = "16-mixed" # The precision for training, see lightning docs for supported types
        accelerator: str = "gpu" # Should be gpu, do not touch
        strategy: str = "ddp_find_unused_parameters_true" # Do not touch
        gradient_clip_val: float = 0.0 # Additional gradient clipping, default is lightning defaults
        gradient_clip_algorithm: str = "norm" # Additional gradient clipping, default is lightning defaults

    @dataclass_json
    @dataclass
    class DataLoaderOptions:
        image_dir: str  # Absolute path to the directory with the images
        mask_dir: str  # Absolute path to the directory containing the tissue masks
        train_csv_path: str  # Path to the csv file. Must be non-empty if stage = "fit"
        val_csv_path: str  # Path to val csv, must be non-empty if stage is one of ["fit", "validate"]
        test_csv_path: str  # Path to test csv, must be non-empty is stage="test"
        mask_suffix: str = "_tissue"  # the suffix for mask tissues e.g. tumor_069_<mask_suffix>.tif
        filetype: str = ".tif"
        num_workers: int = 5
        image_size: int = 8192  # represents image size if variable_input_shape=False, else the maximum image size
        variable_input_shapes: bool = True
        read_level: int = 3  # the level of the tif file (0 is highest resolution)
        transform: Optional[Any] = field(
            default=None, metadata=config(encoder=encode_augmentations, exclude=lambda x: x is None)
        )  # populated in __post_init. only when stage = fit
        tile_stride: Optional[Any] = None  # Calculated dynamically later on
        network_output_stride: Optional[Any] = None  # Calculated later when model is instantiated

    @dataclass_json
    @dataclass
    class MiscOptions:
        stage: str = "fit"  # fit, validation, test, or predict
        ckp_path: str = ""  # the name of the ckp file within the default_save_dir
        resume: bool = True  # Whether to resume training from the last/best epoch
        init_with_mean: bool = False  # If true: warm-start with glmm weights under mean-field approximation
        write_level: int = 2
        wandb_project_name: str = "placeholder"
        wandb_api_key: str = "" # api key to automatically log in


    model_options: ModelOptions
    streaming_options: StreamingOptions
    trainer_options: TrainerOptions
    dataloader_options: DataLoaderOptions
    misc_options: MiscOptions

    def __post_init__(self):
        # Setting the transform based on the stage, moved from DataLoaderOptions to here.
        if self.misc_options.stage == "fit":
            self.dataloader_options.transform = augmentations
        else:
            self.dataloader_options.transform = None

    @classmethod
    def from_args(cls):
        """
        Create a dataclass using argparse. Boolean variables defaults can be changed by appending "no_" to the variable
        name. E.g. if option2 = True, then setting it to False would be equal to --no_option2
        Returns
        -------

        """
        arg_groups: Dict[str, Any] = {
            "Model Options": cls.ModelOptions,
            "Streaming Options": cls.StreamingOptions,
            "Trainer Options": cls.TrainerOptions,
            "Dataloader Options": cls.DataLoaderOptions,
            "Misc Options": cls.MiscOptions,
        }

        parser = argparse.ArgumentParser(description="My Data Class Options")
        for group_name, nested_class in arg_groups.items():
            arg_group = parser.add_argument_group(group_name)
            for field in fields(nested_class):
                # if explicit setting of bools is required, delete these if statement, and keep the else part.

                if field.type == bool:
                    if field.default is True:
                        arg_group.add_argument(
                            f"--no_{field.name}",
                            action="store_false",
                            dest=field.name,
                            help=f"Disable {field.name.replace('_', ' ')}",
                        )
                    else:
                        arg_group.add_argument(
                            f"--{field.name}",
                            action="store_true",
                            dest=field.name,
                            help=f"Enable {field.name.replace('_', ' ')}",
                        )
                else:

                    # Detect if field.type is a union of int and None, adjust accordingly
                    field_type = field.type
                    if isinstance(field_type, types.UnionType) and type(None) in field_type.__args__ and int in field_type.__args__:
                        arg_type = cls.optional_int
                    else:
                        arg_type = field_type


                    default = field.default if field.default != field.default_factory else None
                    arg_group.add_argument(
                        f"--{field.name}", type=arg_type, default=default, help=field.name.replace("_", " ")
                    )

        args = parser.parse_args()

        kwargs = {}
        for group_name, nested_class in arg_groups.items():
            nested_args = {field.name: getattr(args, field.name) for field in fields(nested_class)}
            kwargs[group_name.replace(" ", "_").lower()] = nested_class(**nested_args)

        return cls(**kwargs)

    # Custom type conversion function
    def optional_int(value: str) -> Union[int, None]:
        """Converts input to an integer or None."""
        if value.lower() in ["none", "null"]:
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    def to_flat_dict(self):
        flat_dict = {}
        for nested_class_name in dir(self):
            nested_class = getattr(self, nested_class_name)
            if isinstance(
                nested_class,
                (
                    self.ModelOptions,
                    self.StreamingOptions,
                    self.TrainerOptions,
                    self.DataLoaderOptions,
                    self.MiscOptions,
                ),
            ):
                flat_dict.update(vars(nested_class))
        return flat_dict


if __name__ == "__main__":
    # Usage
    my_data = StreamingCLAMOptions.from_args()

    # Accessing attributes
    print(my_data.model_options.option1)
    print(my_data.trainer_options.trainer_option1)
    print(my_data.dataloader_options.data_loader_option1)
    print(my_data.misc_options.misc_option1)

    flat_options = my_data.to_flat_dict()
    # Accessing attributes
    for key, value in flat_options.items():
        print(f"{key}: {value}")



