import torch
import numpy as np
import pandas as pd

from modules.base import BaseModel

from lightstream.models.resnet.resnet import split_resnet
from applications.streamingclam.clam import CLAM_MB, CLAM_SB

from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50

from torchmetrics import Metric

from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored


class HazardRatio(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.add_state("output", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.cph_model = CoxPHFitter()

    def update(self, output: Tensor, event: Tensor, time: Tensor):
        self.event.append(event)
        self.output.append(output)
        self.time.append(time)

    def compute(self):
        output = np.array([x.cpu().numpy() for x in self.output]).flatten()
        event = np.array([x.cpu().numpy() for x in self.event]).flatten()
        time = np.array([x.cpu().numpy() for x in self.time]).flatten()

        df = pd.DataFrame({"output": output, "event": event, "time": time})

        # documentation on lightning is all over the place
        # but no docs on how to call sanity check compute when more than 2 batches have been processed
        if len(df) > 2:
            self.cph_model.fit(df=df, duration_col="time", event_col="event")
            return torch.as_tensor(self.cph_model.hazard_ratios_["output"])
        else:
            return torch.as_tensor(0)


class Concordance(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.add_state("output", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")

    def update(self, output: Tensor, event: Tensor, time: Tensor):
        self.event.append(event)
        self.output.append(output)
        self.time.append(time)

    def compute(self):
        self.event = np.array([x.cpu().numpy() for x in self.event]).flatten()
        self.output = np.array([x.cpu().numpy() for x in self.output]).flatten()
        self.time = np.array([x.cpu().numpy() for x in self.time]).flatten()

        if len(self.event) > 2:
            c_index, concordant, disconcordant, tied_risk, tied_time = concordance_index_censored(
                self.event, self.time, self.output
            )
        else:
            return torch.as_tensor(0)
        return torch.as_tensor(c_index)


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
        *args,
        **kwargs,
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
        elif self.encoder in ("resnet18", "resnet34"):
            return [512, 512, 256]

    def configure_model(self):
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


class StreamingCLAM(BaseModel):
    model_choices = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}

    def __init__(
        self,
        encoder: str,
        tile_size: int,
        loss_fn: torch.nn.functional,
        branch: str,
        n_classes: int,
        max_pool_kernel: int = 0,
        stream_max_pool_kernel: bool = False,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
        *args,
        **kwargs,
    ):
        self.stream_maxpool_kernel = stream_max_pool_kernel
        self.max_pool_kernel = max_pool_kernel
        self.instance_eval = instance_eval
        self.return_features = return_features
        self.attention_only = attention_only

        if self.max_pool_kernel < 0:
            raise ValueError(f"max_pool_kernel must be non-negative, found {max_pool_kernel}")
        if self.stream_maxpool_kernel and self.max_pool_kernel == 0:
            raise ValueError(f"stream_max_pool_kernel cannot be True when max_pool_kernel=0")

        assert encoder in list(StreamingCLAM.model_choices.keys())

        # Define the streaming network and head
        network = StreamingCLAM.model_choices[encoder](weights="IMAGENET1K_V1")
        stream_net, _ = split_resnet(network)
        head = CLAMConfig(encoder=encoder, branch=branch, n_classes=n_classes, **kwargs).configure_model()

        # At the end of the ResNet model, reduce the spatial dimensions with additional max pool
        self.ds_blocks = None
        if self.max_pool_kernel > 0:
            if self.stream_maxpool_kernel:
                stream_net = self.add_maxpool_layers(stream_net)
                super().__init__(stream_net, head, tile_size, loss_fn, *args, **kwargs)
            else:
                ds_blocks, head = self.add_maxpool_layers(head)
                super().__init__(stream_net, head, tile_size, loss_fn, *args, **kwargs)
                self.ds_blocks = ds_blocks
        else:
            super().__init__(stream_net, head, tile_size, loss_fn, *args, **kwargs)

        self.train_hazard_ratio = HazardRatio()
        self.val_hazard_ratio = HazardRatio()

        self.train_index = Concordance()
        self.val_index = Concordance()

    def add_maxpool_layers(self, network):
        ds_blocks = torch.nn.Sequential(torch.nn.MaxPool2d((self.max_pool_kernel, self.max_pool_kernel)))

        if self.stream_maxpool_kernel:
            return torch.nn.Sequential(network, ds_blocks)
        else:
            return ds_blocks, network

    def forward_head(
        self, fmap, mask=None, instance_eval=False, label=None, return_features=False, attention_only=False
    ):
        batch_size, num_features, h, w = fmap.shape

        if self.ds_blocks is not None:
            fmap = self.ds_blocks(fmap)

        # Mask background, can heavily reduce inputs to clam network
        if mask is not None:
            fmap = torch.masked_select(fmap, mask)
            del mask

        # Put everything back together into an array [channels, #unmasked_pixels]
        # Change dimensions from [batch_size, C, H, W] to [batch_size, C, H * W]
        fmap = torch.reshape(fmap, (num_features, -1)).transpose(0, 1)

        if self.attention_only:
            return self.head(fmap, label=None, instance_eval=False, attention_only=self.attention_only)

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.head(
            fmap,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )

        return logits, Y_prob, Y_hat, A_raw, instance_dict

    def forward(self, x, mask=None):
        if len(x) == 2:
            image, mask = x
        else:
            image = x

        fmap = self.forward_streaming(image)
        out = self.forward_head(
            fmap, mask=mask, return_features=self.return_features, attention_only=self.attention_only
        )
        return out

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        if len(batch) == 5:
            image, mask, label, follow_up, event = batch
        else:
            image, label, follow_up, event = batch
            mask = None

        self.image = image
        self.str_output = self.forward_streaming(image)
        self.str_output.requires_grad = self.training

        # Can only be changed when streaming is enabled, otherwise not a leaf variable
        # if self.use_streaming:
        #    self.str_output.requires_grad = True

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward_head(
            self.str_output,
            mask=mask,
            instance_eval=self.instance_eval,
            label=label if self.instance_eval else None,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )

        loss = self.loss_fn(logits.flatten(), label)

        self.train_hazard_ratio(logits.flatten().detach(), label.detach(), follow_up.detach())
        self.log("train_hazard_ratio", self.train_hazard_ratio, on_epoch=True)

        self.train_index(logits.flatten().detach(), event.detach(), label.detach())
        self.log("train_index", self.train_index, on_epoch=True)

        self.log_dict({"entropy loss": loss.detach()}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, label, follow_up, event = self._shared_eval_step(batch, batch_idx)

        # Should update and clear automatically, as per
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        # https: // lightning.ai / docs / pytorch / stable / extensions / logging.html
        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        self.val_hazard_ratio(y_hat.flatten(), label.detach(), follow_up.detach())
        self.log("val_hazard_ratio", self.val_hazard_ratio, on_epoch=True)

        self.val_index(y_hat.flatten(), event.detach(), label.detach())
        self.log("val_index", self.val_index, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss, y_hat, label, follow_up, event = self._shared_eval_step(batch, batch_idx)

        self.test_acc(torch.argmax(y_hat, dim=1))
        self.test_auc(torch.sigmoid(y_hat)[:, 1])

        metrics = {"test_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        if len(batch) == 5:
            image, mask, label, follow_up, event = batch
        else:
            image, label, follow_up, event = batch
            mask = None

        y_hat = self.forward(image)[0].detach()
        loss = self.loss_fn(y_hat.flatten(), label).detach()

        return loss, y_hat, label.detach(), follow_up.detach(), event.detach()


if __name__ == "__main__":
    model = StreamingCLAM(
        "resnet18",
        tile_size=1600,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=4,
        max_pool_kernel=8,
    )
