import torch

from lightstream.modules.imagenet_template import ImageNetClassifier

from lightstream.models.resnet.resnet import split_resnet
from streamingclam.models.clam import CLAM_MB, CLAM_SB

from torchvision.models import resnet18, resnet34, resnet50
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, AUROC


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


class StreamingCLAM(ImageNetClassifier):
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
        train_streaming_layers: bool = False,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
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
        head = CLAMConfig(encoder=encoder, branch=branch, n_classes=n_classes).configure_clam()

        # At the end of the ResNet model, reduce the spatial dimensions with additional max pool
        self._get_streaming_options(**kwargs)

        self.ds_blocks = None
        if self.max_pool_kernel > 0:
            if self.stream_maxpool_kernel:
                stream_net = self.add_maxpool_layers(stream_net)
                super().__init__(
                    stream_net,
                    head,
                    tile_size,
                    loss_fn,
                    train_streaming_layers=train_streaming_layers,
                    **self.streaming_options,
                )
            else:
                ds_blocks, head = self.add_maxpool_layers(head)
                super().__init__(
                    stream_net,
                    head,
                    tile_size,
                    loss_fn,
                    train_streaming_layers=train_streaming_layers,
                    **self.streaming_options,
                )
                self.ds_blocks = ds_blocks
        else:
            super().__init__(
                stream_net,
                head,
                tile_size,
                loss_fn,
                train_streaming_layers=train_streaming_layers,
                **self.streaming_options,
            )

        # Define metrics
        metrics = MetricCollection(
            [
                Accuracy(task="binary", num_classes=n_classes),
                AUROC(task="binary", num_classes=n_classes),
            ]
        )

        self.train_acc = Accuracy(task="binary", num_classes=n_classes)
        self.train_auc = AUROC(task="binary", num_classes=n_classes)

        self.val_acc = Accuracy(task="binary", num_classes=n_classes)
        self.val_auc = AUROC(task="binary", num_classes=n_classes)

        self.test_acc = Accuracy(task="binary", num_classes=n_classes)
        self.test_auc = AUROC(task="binary", num_classes=n_classes)

        self.test_outputs = []

    def add_maxpool_layers(self, network):
        ds_blocks = torch.nn.Sequential(
            torch.nn.MaxPool2d((self.max_pool_kernel, self.max_pool_kernel), ceil_mode=True)
        )

        if self.stream_maxpool_kernel:
            return torch.nn.Sequential(network, ds_blocks)
        else:
            return ds_blocks, network

    def forward_head(
        self,
        fmap,
        mask=None,
        instance_eval=False,
        label=None,
        return_features=False,
        attention_only=False,
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
            return self.head(
                fmap,
                label=None,
                instance_eval=False,
                attention_only=self.attention_only,
            )

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.head(
            fmap,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )

        return logits, Y_prob, Y_hat, A_raw, instance_dict

    def forward(self, image, mask=None):
        fmap = self.forward_streaming(image)
        out = self.forward_head(
            fmap,
            mask=mask,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )
        return out

    def training_step(self, batch, batch_idx: int, *args, **kwargs):
        image, mask, label, fname = batch[0]["image"], batch[0]["mask"], batch[1], batch[2]
        image = image.to("cpu")

        self.image = image
        self.str_output = self.forward_streaming(image)
        self.str_output.requires_grad = self.training

        logits, Y_prob, Y_hat, A_raw, instance_dict = self.forward_head(
            self.str_output,
            mask=mask,
            instance_eval=self.instance_eval,
            label=label if self.instance_eval else None,
            return_features=self.return_features,
            attention_only=self.attention_only,
        )

        loss = self.loss_fn(logits, label)
        self.train_acc.update(torch.argmax(logits, dim=1).detach(), label.detach())
        self.train_auc.update(torch.sigmoid(logits)[:, 1].detach(), label.detach())

        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc_epoch", self.train_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("train_auc_epoch", self.train_auc.compute(), prog_bar=True, sync_dist=True)

        self.train_acc.reset()
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        loss, y_hat, label, fname = self._shared_eval_step(batch, batch_idx)
        del batch

        self.val_acc.update(torch.argmax(y_hat, dim=1).detach(), label)
        self.val_auc.update(torch.sigmoid(y_hat)[:, 1], label)

        # Should update and clear automatically, as per
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        # https: // lightning.ai / docs / pytorch / stable / extensions / logging.html
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.val_acc.compute(), sync_dist=True)
        self.log("val_auc_epoch", self.val_auc.compute(), sync_dist=True)

        self.val_acc.reset()
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        loss, y_hat, label, fname = self._shared_eval_step(batch, batch_idx)
        probs = torch.sigmoid(y_hat)

        self.test_acc(torch.argmax(y_hat, dim=1), label)
        self.test_auc(probs[:, 1], label)

        metrics = {
            "test_acc": self.test_acc,
            "test_auc": self.test_auc,
            "test_loss": loss,
        }
        self.log_dict(metrics, prog_bar=True)
        outputs = [
            fname,
            label.cpu().numpy()[0],
            torch.argmax(y_hat, dim=1).cpu().numpy()[0],
            probs[:, 0][0].cpu().numpy(),
            probs[:, 1][0].cpu().numpy(),
        ]

        self.test_outputs.append(outputs)

        return outputs

    def _shared_eval_step(self, batch, batch_idx):
        image, mask, label, fname = batch[0]["image"], batch[0]["mask"], batch[1], batch[2]
        image = image.to("cpu")

        y_hat = self.forward(image, mask=mask)[0].detach()
        loss = self.loss_fn(y_hat, label)
        return loss, y_hat, label.detach(), fname

    def _get_streaming_options(self, **kwargs):
        """Set streaming defaults, but overwrite them with values of kwargs if present."""

        # We need to add torch.nn.Batchnorm to the keep modules, because of some in-place ops error if we don't
        # https://discuss.pytorch.org/t/register-full-backward-hook-for-residual-connection/146850
        streaming_options = {
            "verbose": True,
            "copy_to_gpu": False,
            "statistics_on_cpu": True,
            "normalize_on_gpu": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "add_keep_modules": [torch.nn.BatchNorm2d],
        }
        self.streaming_options = {**streaming_options, **kwargs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.params, lr=1e-4, weight_decay=1e-5)
        return opt

    def backward(self, loss):
        loss.backward()
        # del loss
        # Don't call this>? https://pytorch-lightning.readthedocs.io/en/1.5.10/guides/speed.html#things-to-avoid
        torch.cuda.empty_cache()
        if self.train_streaming_layers:
            self.backward_streaming(self.image, self.str_output.grad)
        del self.str_output, self.image


if __name__ == "__main__":
    model = StreamingCLAM(
        "resnet18",
        tile_size=1600,
        loss_fn=torch.nn.functional.cross_entropy,
        branch="sb",
        n_classes=4,
        max_pool_kernel=8,
    )
