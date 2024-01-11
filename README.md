# pathology-streamingclam
Lightstream implementation of the StreamingCLAM model


## Running StreamingCLAM

To run the streamingclam, first git clone the repository, and optionally add it to your PYTHONPATH if needed:

```
git clone git@github.com:DIAGNijmegen/pathology-streamingclam.git
```

Then, run the main.py with at least the required options. For example, to train the streamingCLAM model starting from ImageNet weights:

``` bash
python main.py --image_path <path_to_images> --train_csv <path_to_csv_file> --val_csv <path_to_val_csv> --default_save_dir <path_to_save_results>
```

Or, if you have tissue masks, you can use them by passing them into main.py as such:

``` bash
python main.py --image_path <path_to_images> --mask_path <path_to_tissue_masks> --mask_suffix <mask_suffix> --train_csv <path_to_csv_file> --val_csv <path_to_val_csv> --default_save_dir <path_to_save_results>
```

We assume that the masks have the same names as the images, but can have an additional postfix, e.g. if the name of the image is `tumor_001.tif`, then the mask can be called `tumor_001_tissue.tif`, and you must add `mask_suffix _tissue` as an additional argument. 

## Structure of the csv_files
The structure of the csv files must be as follows. A concrete example is posted under the data_splits folder. Column headers are not required. However, it is important that the first column has the image names, and the second column contains the labels as integer numbers.


```
slide_id,label
TRAIN_1,1
TRAIN_2,1
TRAIN_3,0
```

### Options

There are quite some options (disable boolean options by prepending with `no_`, so e.g., `no_use_augmentations`):

| Required options | Description |
| ---:         |     :---      |
| `image_path: str` | The name of the current experiment, used for saving checkpoints. |
| `default_save_dir: str` | The directory where logs and checkpoints are stored|
| `mask_path: int` | The number of classes in the task. |
| `train_csv: str` | The filenames (without extension) and labels of train set. |
| `val_csv: str` | The filenames (without extension) and labels of validation or test set. |
| `test_csv: str` | The directory where the images reside. |
| `mask_suffix: str` = "_tissue:" | the suffix for mask tissues e.g. tumor_069_<mask_suffix>.tif. Default is "_tissue"|
| `mode: str = "fit"`  | fit, validation, test or predict, default is fit|
| `unfreeze_streaming_layers_at_epoch: int = 15` | The epoch to unfreeze and train the entire network. Default is 20
| **Trainer options** | |
| `num_epochs: int = 50` | The maximum number of epochs to train, default is 50|
| `strategy: str = "ddp_find_unused_parameters_true"` | The training strategy. Suggested to use 'auto' for 1 gpu, and "ddp_find_unused_parameters_true" for multiple gpu's.|
| `ckp_path: str = ""` | # the name fo the ckp file within the default_save_dir|
| `resume: bool = True` | # Whether to resume training from the last/best epoch|
| `grad_batches: int = 2` | # Gradient accumulation: the amount of batches before optimizer step. Default is 2|
| `num_gpus: int`| The number of devices to use |
| `precision: str = "16-mixed"`| The precision during training. Recommended to use "16-mixed", "16-true", "bf16-mixed", "bf16-true"|
| **StreamingCLAM options** | |
|`encoder: str = "resnet34"` | "resnet18", "resnet34", "resnet50". Default is "resnet34"|
|`branch: str = "sb"`  |single branch "sb" or multi-branch "mb" clam head. Default is "sb"|
|`max_pool_kernel: int = 8`| The size of the max pool kernel and stride at the end of the resnet backbone|
|`num_classes: int = 2`||
|`loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()`| The loss function for the CLAM classification head. Default is torch.nn.CrossEntropyLoss()|
|`instance_eval: bool = False`| Whether to use CLAM's instance_eval. Default is "False"|
|`return_features: bool = False`| Return the features "M" of the CLAM attention "M = torch.mm(A, h)" where A is the normalized attention map and h is the input features of the resnet  |
|`attention_only: bool = False`| Whether to only return the attention map A_raw. In this case, the attention is unnormalized (no softmax)|
|`stream_max_pool_kernel: bool = False`| If set to true, will incorporate the max_pool_kernel into the streaming network. This will save memory, at the cost of speed|
|`learning_rate: float = 2e-4`  | The learning rate for training the CLAM head|
| **Streaming options** | |
|`tile_size: int = 9984`| The tile size for the streaming network. Higher uses more GPU memory, but also speeds up computations|
|`statistics_on_cpu: bool = True`| Calculate the streaming network statistics on the cpu. Default is True|
|`verbose: bool = True`|Verbose streaming network output. Default is True|
|`train_streaming_layers: bool = False`| Whether to train the streaming layers. Default is False, will set to true at "unfreeze_streaming_layers_at_epoc"|
|`normalize_on_gpu: bool = True`| Do not change this!: Normalize with imagenet weights on the gpu. Default is True|
|`copy_to_gpu: bool = False`| Copies the entire image to the gpu. Recommended False if image > 16000x16000|
| **Dataloader options** | |
|`image_size: int = 65536`| represents image size if variable_input_shape=False, else the maximum image size|
|`variable_input_shapes: bool = True`| If true, will not pad/crop images to a uniform size specified by image_size. Can save RAM and speed up training/inference|
|`filetype: str = ".tif"`| The filetype of the images. Default is ".tif"|
|`read_level: int = 1` |  The level of the tif file (0 is highest resolution)|
|`num_workers: int = 3`| The amount of workers per gpu. Default is 3. |
|`use_augmentations: bool = True`| Whether to use augmentations specified in the file data.dataset.py. Default is True|



