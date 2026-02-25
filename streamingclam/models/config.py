from streamingclam.models.clam import CLAM_MB, CLAM_SB
import torch
import torch.nn as nn

from lightstream.models.resnet import StreamingResNet
from lightstream.models.convnext.convnext import StreamingConvNext

# Streamingclam works with resnets, can be extended to other encoders if needed
class CLAMConfig:
    def __init__(
        self,
        encoder: str,
        branch: str,
        n_classes: int = 2,
        gate: bool = True,
        use_dropout: bool = False,
        k_sample: int = 8,
        instance_loss_fn: torch.nn = torch.nn.CrossEntropyLoss,
        subtyping=False,
    ):
        self.branch = branch
        self.encoder = encoder
        self.n_classes = n_classes
        self.size = self.configure_size()

        self.gate = gate
        self.use_dropout = use_dropout
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping

    def configure_size(self):
        if self.encoder == "resnet50":
            return [2048, 512, 256]
        elif self.encoder == "resnet39":
            return [1024, 512, 256]
        elif self.encoder in ("resnet18", "resnet34"):
            return [512, 512, 256]

    def configure_clam(self):
        # size args original: self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        if self.branch == "sb":
            print("Loading CLAM with single branch \n")
            return CLAM_SB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        elif self.branch == "mb":
            print("Loading CLAM with multiple branches \n")

            return CLAM_MB(
                gate=self.gate,
                size=self.size,
                dropout=self.use_dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=self.instance_loss_fn(),
                subtyping=self.subtyping,
            )
        else:
            raise NotImplementedError(
                f"branch must be specified as single-branch " f"'sb' or multi-branch 'mb', not {self.branch}"
            )


def configure_pooling_layer(pooling_layer: str | None = None, pooling_kernel: int = 0):
    assert pooling_layer in ["avgpool", "maxpool", "none"]
    if pooling_layer == "maxpool":
        return nn.MaxPool2d((pooling_kernel, pooling_kernel), ceil_mode=True)
    elif pooling_layer == "avgpool":
        return nn.AvgPool2d((pooling_kernel, pooling_kernel), ceil_mode=True)
    else:
        return nn.Identity()


def configure_backbone(**kwargs):
    encoder = kwargs.pop("encoder")
    tile_size = kwargs.pop("tile_size")
    stream_pooling_kernel = kwargs.pop("stream_pooling_kernel")
    pooling_layer = configure_pooling_layer(kwargs.pop("pooling_layer"), kwargs.pop("pooling_kernel"))

    model_map = {
        **{name: StreamingResNet for name in StreamingResNet.get_model_names()},
        **{name: StreamingConvNext for name in StreamingConvNext.get_model_names()},
    }

    assert encoder in model_map, f"Unsupported encoder: {encoder}"
    model_cls = model_map[encoder]

    model = model_cls(
        encoder,
        tile_size,
        additional_modules=pooling_layer if stream_pooling_kernel else None,
        **kwargs,
    )

    pooling_layer = nn.Identity() if stream_pooling_kernel else pooling_layer
    return model, pooling_layer