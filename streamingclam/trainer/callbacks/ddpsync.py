"""
This file contains callbacks that monitors the ddp processes across gpu's during training.
It checks for optimizer states and parameter gradients
If ranks diverge (i.e. weights differ) then it will proceed to print warnings that states have diverged

It is not needed to run streaming, but it is a file for additional debugging.
"""

import torch
import torch.distributed as dist
from lightning.pytorch import Callback
import copy


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Ensures all optimizer state tensors are on the same device for reliable comparison.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_optimizer_state_dict(optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    """
    Moves optimizer state to device and returns its state dict.
    """
    move_optimizer_state_to_device(optimizer, device)
    return optimizer.state_dict()



def flatten_to_cpu(state_dict: dict) -> dict:
    def move(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        elif isinstance(v, dict):
            return {k: move(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [move(i) for i in v]
        elif isinstance(v, tuple):
            return tuple(move(i) for i in v)
        return v

    return move(copy.deepcopy(state_dict))


def safe_compare(a, b) -> bool:
    """Recursively compare a and b, safely handling tensors."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.equal(b)
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(safe_compare(a[k], b[k]) for k in a)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(safe_compare(x, y) for x, y in zip(a, b))
    elif isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(safe_compare(x, y) for x, y in zip(a, b))
    else:
        return a == b


def check_optimizer_state_sync(optimizer: torch.optim.Optimizer, device: torch.device, strict: bool = False) -> None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    local_state = flatten_to_cpu(optimizer.state_dict())

    object_list = [local_state] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(object_list, src=0)

    ref_state = object_list[0]
    if dist.get_rank() != 0 and not safe_compare(local_state, ref_state):
        msg = f"[RANK {dist.get_rank()}] Optimizer state differs from rank 0!"
        print(msg)
        if strict:
            raise RuntimeError(msg)



def verify_optimizer_lr_sync(optimizer, device):
    if not dist.is_initialized():
        return

    lr_tensor = torch.tensor([g["lr"] for g in optimizer.param_groups], device=device)
    gathered = [torch.zeros_like(lr_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, lr_tensor)

    if not all(torch.allclose(lr_tensor, g, atol=1e-8) for g in gathered):
        raise RuntimeError(f"Learning rates differ across ranks! This breaks DDP sync.")

def verify_param_ids_sync(optimizer, device):
    if not dist.is_initialized():
        return

    id_tensor = torch.tensor([id(p) % 2**24 for g in optimizer.param_groups for p in g["params"]], device=device)
    gathered = [torch.zeros_like(id_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, id_tensor)

    if not all(torch.equal(id_tensor, g) for g in gathered):
        raise RuntimeError("Parameter lists in optimizer differ across ranks.")

def verify_grads(optimizer, device):
    # Optional: Check requires_grad flags
    requires_grad_tensor = torch.tensor(
        [int(p.requires_grad) for g in optimizer.param_groups for p in g["params"]],
        dtype=torch.uint8,
        device=device
    )
    gathered_flags = [torch.zeros_like(requires_grad_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_flags, requires_grad_tensor)
    if not all(torch.equal(requires_grad_tensor, other) for other in gathered_flags):
        raise RuntimeError(f"[Rank {dist.get_rank()}] Parameter requires_grad mismatch between ranks")


def verify_state_keys(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    # Convert param keys to consistent ids for comparison
    if len(optimizer.state) > 0:
        state_keys = torch.tensor(
            sorted([id(k) % (2**24) for k in optimizer.state.keys()]),
            dtype=torch.int32,
            device=device,
        )
    else:
        state_keys = torch.tensor([0], dtype=torch.int32, device=device)

    max_len = state_keys.shape[0]
    gathered = [torch.zeros(max_len, dtype=torch.int32, device=device) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, state_keys)

    if not all(torch.equal(state_keys, g) for g in gathered):
        raise RuntimeError(f"[Rank {torch.distributed.get_rank()}] Optimizer state keys mismatch across ranks")



class OptimizerStateSyncCheck(Callback):
    """
    Checks that all optimizer states are synchronized across ranks during DDP training.
    Can help debug training divergence.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def on_train_epoch_start(self, trainer, pl_module):
        if not dist.is_initialized():
            return

        # Optionally skip check after resume epoch
        if trainer.current_epoch == 0:
            return

        optimizer = trainer.optimizers[0]
        device = pl_module.device
        check_optimizer_state_sync(optimizer, device, strict=self.strict)
        verify_optimizer_lr_sync(optimizer, device)
        #verify_param_ids_sync(optimizer, device)
        #verify_state_keys(optimizer, device)
        verify_grads(optimizer, device)