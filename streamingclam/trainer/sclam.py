import logging
import pandas as pd
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch import Tensor

from lightning.pytorch.utilities.types import STEP_OUTPUT

from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, BinaryCalibrationError, MulticlassCalibrationError

from lightstream import LightningStreamingModule
from streamingclam.models.config import CLAMConfig, configure_backbone
from streamingclam.trainer.callbacks.streamingwriter import StreamingHeatmapWriter

class StreamingCLAM(LightningStreamingModule):

    def __init__(
        self,
        branch: str,
        num_classes: int,
        gate: bool = True,
        use_dropout: bool = True,
        instance_eval: bool = False,
        bag_weight: float = 1.0,
        k_sample: int = 8,
        subtyping: bool = False,
        initial_lr: float = 2e-4,
        finetune_lr: float = 2e-5,
        accumulate_grad_batches: int = 1,
        unfreeze_epoch: int = 15,
        **kwargs,
    ):
        self.save_hyperparameters(ignore=['loss_fn', 'instance_loss_fn'])
        encoder = kwargs.get("encoder")
        model, pooling_layer = configure_backbone(**kwargs)

        super().__init__(model)

        self.pooling_layer = pooling_layer  # Can be None if chosen as None, or if it's part of the streaming network

        config = CLAMConfig(encoder, branch, num_classes, gate, use_dropout, k_sample, nn.CrossEntropyLoss, subtyping)
        self.clam = config.configure_clam()

        self.num_classes = num_classes

        self.initial_lr = initial_lr
        self.finetune_lr = finetune_lr
        self.accumulate_grad_batches = accumulate_grad_batches
        self.unfreeze_epoch = unfreeze_epoch
        self.loss_fn = nn.CrossEntropyLoss()
        self.instance_eval = instance_eval
        self.bag_weight = bag_weight
        self.test_outputs = []


        # Will be handled by the trainer
        self._init_metrics()
        self.train_streaming_layers = False  # Adjusted later
        self.automatic_optimization = False

        #  Adjusted at unfreeze epoch
        for param in self.stream_network.parameters():
            param.requires_grad = False

    def create_output_dirs(self, fname):
        # these class variables changes with each new image in the test set batch being processed
        self.heatmap_dir = Path(self.trainer.default_root_dir + "/heatmaps/" + fname)

        if not self.heatmap_dir.exists():
            self.heatmap_dir.mkdir(parents=True, exist_ok=True)

    def _init_metrics(self):
        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
                "auc": AUROC(task="multiclass", num_classes=self.num_classes),
                "balanced_acc": Accuracy(task="multiclass", num_classes=self.num_classes, average="macro"),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

    def gather_loss_metrics(self, loss_dict: dict, prefix: str | None = None) -> dict:
        """
        Gather loss components and put them in individual metrics.
        Parameters
        ----------
        loss_dict : dict
            Dictionary with loss components from sclam
        prefix : str

        Returns
        -------
        result_dict : dict
            Dictionary with loss components

        """
        result_dict = {
            f"{prefix}total_loss": loss_dict["total_loss"],
            f"{prefix}ll_loss": loss_dict["ll"],
        }

        if 'instance_loss' in loss_dict.keys():
            result_dict[f"{prefix}inst_loss"] = loss_dict["instance_loss"]

        return result_dict

    def _log_lrs(self, batch_size=1):
        opt = self.trainer.optimizers[0]
        self.log(
            "lr/backbone",
            opt.param_groups[0]["lr"],
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "lr/head", opt.param_groups[1]["lr"], prog_bar=True, sync_dist=True, on_epoch=True, batch_size=batch_size
        )

    def loss(self, logits, results_dict, label):
        """ gather loss components for clam"""

        loss = self.bag_weight * self.loss_fn(logits, label[0])
        total_loss = loss
        # gather loss objects
        ldict = {}
        if 'instance_loss' in results_dict.keys():
            instance_loss = results_dict["instance_loss"] * (1-self.bag_weight)
            total_loss = total_loss + instance_loss
            ldict["instance_loss"] = instance_loss.item()

        ldict = {'total_loss': total_loss.item(), 'll': loss.item()}
        return total_loss, ldict

    def forward(self, image, mask=None, label=None, instance_eval=None):
        fmap = self.forward_streaming(image)
        return self.forward_clam(fmap, mask=mask, label=label, instance_eval=instance_eval)

    def forward_clam(self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        label: torch.Tensor = None,
        instance_eval: bool = False,
    ):

        features = self.pooling_layer(features) if self.pooling_layer else features

        channels = features.shape[1]
        if mask is not None and not torch.all(~mask):
            features = torch.masked_select(features, mask)  # Mask out non-tissue areas with the tissue background mask
            del mask

        # Put everything back together into an array [channels, #unmasked_pixels]
        # Change dimensions from [batch_size, C, H, W] to [batch_size, C, H * W]
        features = torch.reshape(features, (channels, -1)).transpose(0, 1)

        return self.clam.forward(features, label=label, instance_eval=instance_eval)


    def forward_streaming(self, x):
        return self.stream_network.forward(x)


    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.unfreeze_epoch:
            print("Training streaming layers")

        if self.current_epoch >= self.unfreeze_epoch:
            # >= here, when resuming should put this on
            self.unfreeze_streaming_network()
            self.train_streaming_layers = True

    def on_train_epoch_end(self) -> None:
        self.lr_schedulers().step()

    def training_step(self, batch, batch_idx):

        image, label = batch.image, batch.label
        mask = getattr(batch, "mask", None)

        features = self.forward_streaming(image)
        features.requires_grad = True

        logits, Y_prob, Y_hat, A_raw, results_dict = self.forward_clam(features, mask, label, self.instance_eval)

        loss, ldict = self.loss(logits, results_dict, label)
        loss = loss / self.accumulate_grad_batches

        self._backward_streaming(loss, image, features)
        self._distribute_gradients()
        self._optimizer_step_if_needed(batch_idx)

        loss_dict = self.gather_loss_metrics(ldict, prefix="train/") # converts losses to loggable train/val/test metrics

        # Logging happens at batch size
        self.train_metrics(Y_prob, label[0].long(), sync_dist=True)
        self.log_dict(self.train_metrics, on_epoch=True, sync_dist=True, batch_size=1)
        self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self._log_lrs()

    def _backward_streaming(self, loss: Tensor, image: Tensor, features: Tensor) -> None:
        if self.trainer.num_devices > 1:
            with self.trainer.strategy.model.no_sync():
                self.manual_backward(loss)
        else:
            self.manual_backward(loss)
        if self.train_streaming_layers:
            self.stream_network.backward(image, features.grad)

    def _distribute_gradients(self):
        if self.trainer.num_devices > 1:
            self.sync_gradients(self, log_rank=0)

    def sync_gradients(
        self,
        model: nn.Module,
        *,
        log_rank: int | None = None,
        skip_if_frozen: bool = True,
    ) -> None:
        """
        Synchronize gradients across DDP ranks, injecting dummy grads where missing.
        Also logs the parent module type for each parameter with missing grad.
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Build mapping from param -> parent module
        param_to_module = {}
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                param_to_module[full_name] = module

        for name, param in model.named_parameters():
            if skip_if_frozen and not param.requires_grad:
                continue

            if param.grad is None: # check if/when grads are 0, shouldn't happen, can cause hangs/bugs if reported
                if log_rank is None or rank == log_rank:
                    mod = param_to_module.get(name, None)
                    mod_str = f"{type(mod).__name__}" if mod else "UnknownModule"
                    print(f"[Rank {rank}] Injecting zero grad for: {name} ({mod_str})")
                param.grad = torch.zeros_like(param, device=param.device)

            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    def _optimizer_step_if_needed(self, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        for opt in opts: # accumulate gradients of N batches
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt.step()
                opt.zero_grad()

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, label = batch.image, batch.label
        mask = getattr(batch, "mask", None)

        logits, Y_prob, Y_hat, A_raw, results_dict = self(image, mask, label, self.instance_eval)
        loss, ldict = self.loss(logits, results_dict, label)
        loss_dict = self.gather_loss_metrics(ldict, prefix="val/") # converts losses to loggable train/val/test metrics

        # Logging happens at batch size
        self.val_metrics(Y_prob, label[0].long(), sync_dist=True)
        self.log_dict(self.val_metrics, on_epoch=True, sync_dist=True, batch_size=1)
        self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)


    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        image, label = batch.image, batch.label
        mask = getattr(batch, "mask", None)

        logits, Y_prob, Y_hat, A_raw, results_dict = self(image, mask,label, self.instance_eval)

        loss, ldict = self.loss(logits, results_dict, label)
        loss_dict = self.gather_loss_metrics(ldict, prefix="test/") # converts losses to loggable train/val/test metrics

        self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.test_metrics.update(Y_prob.detach(), label[0].long().detach())

        self.test_outputs.append(
            {
                "slide_name": batch.metadata["filename"],
                "probs": Y_prob.detach().cpu().numpy(),
                "y_hat": Y_hat.detach().cpu().numpy(),
                "label": label.cpu().numpy(),
            }
        )

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:

        # Don't analyze when overwrite=False and result exists, otherwise it will take ages
        for cb in self.trainer.callbacks:
            if isinstance(cb, StreamingHeatmapWriter):
                overwrite = cb.overwrite
                heatmap_dir = self.trainer.default_root_dir / Path("heatmaps") / Path(batch.metadata['filename'])

        if heatmap_dir.exists() and not overwrite:
            return 0,0,0,0 # Dummy return to make the callback not crash

        image, label = batch.image, batch.label
        mask = getattr(batch, "mask", None)

        features = self.forward_streaming(image)  # using self.forward() is expensive, so only stream once
        logits, Y_prob, Y_hat, A_raw, results_dict = self.forward_clam(features, mask)

        return logits, Y_prob, Y_hat, A_raw

    def configure_optimizers(self):
        backbone_params = list(self.stream_network.stream_module.parameters())
        head_params = list(self.clam.parameters())

        # Optimizer setup
        optimizer = torch.optim.RAdam(
            [
                {
                    "params": backbone_params,
                    "lr": self.finetune_lr,
                },  # effective LR = 0 initially via lambda
                {"params": head_params, "lr": self.initial_lr},
            ],
            weight_decay=1e-5,
        )

        def backbone_lambda(epoch: int) -> float:
            return 0.0 if epoch < self.unfreeze_epoch else 1.0  # stays 0 for first 2 epochs

        def head_lambda(epoch: int) -> float:
            return 1.0 if epoch < self.unfreeze_epoch else self.finetune_lr / self.initial_lr

        scheduler = LambdaLR(optimizer, lr_lambda=[backbone_lambda, head_lambda])

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        ]

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

        # Merge outputs
        merged_dict = {}
        for d in self.test_outputs:
            for k, v in d.items():
                merged_dict.setdefault(k, []).extend([v] if isinstance(v, str) else v)

        df = pd.DataFrame(merged_dict)
        csv_name = self.trainer.test_dataloaders.dataset.csv_path.stem
        self._write_test_results(df, csv_name)

    def _write_test_results(self, df: pd.DataFrame, csv_name: str):
        split_cols = pd.DataFrame(df["probs"].tolist())
        split_cols.columns = [f"p_{i}" for i in range(split_cols.shape[1])]
        df = df.drop(columns=["probs"]).join(split_cols)
        df["y_hat"] = df["y_hat"].apply(lambda x: x[0])
        df["label"] = df["label"].apply(lambda x: x[0])

        result_path = Path(self.trainer.default_root_dir) / f"{csv_name}_results.csv"
        df.to_csv(result_path, index=False)
        logging.info(f"Saved results to {result_path}")

        metrics = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in self.trainer.callback_metrics.items()
        }
        metrics_path = Path(self.trainer.default_root_dir) / f"{csv_name}_metrics.csv"
        print("writing metrics to ", metrics_path)
        pd.DataFrame([metrics]).to_csv(metrics_path, mode="a", index=False)

