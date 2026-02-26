import torch
from collections import OrderedDict
from streamingclam.trainer.sclam import StreamingCLAM
from streamingclam.trainer.options import StreamingCLAMOptions

weights_old = torch.load("PATH here")


sd_backbone = weights_old["state_dict_checkpointed"]  # keys like "0.weight", "4.0.conv1.weight"
sd_head     = weights_old["state_dict_net"]           # keys like "head.attention_net..."

new_sd = OrderedDict()

# A) backbone: prefix -> stream_network.stream_module.
for k, v in sd_backbone.items():
    new_sd[f"stream_network.stream_module.{k}"] = v

# B) head: head.* -> clam.*
for k, v in sd_head.items():
    if k.startswith("head."):
        k2 = k[len("head."):]   # drop "head."
    else:
        k2 = k
    new_sd[f"clam.{k2}"] = v

print("Converted tensors:", len(new_sd))
print("Example keys:", list(new_sd.keys())[:10])


def configure_model(options):
    model_options = options.model_options.to_dict()
    streaming_options = options.streaming_options.to_dict()
    return StreamingCLAM(**model_options, **streaming_options)

options = StreamingCLAMOptions.from_args()

lit = configure_model(options)

target = lit.state_dict()

filtered = OrderedDict(
    (k, v) for k, v in new_sd.items()
    if k in target and target[k].shape == v.shape
)

missing, unexpected = lit.load_state_dict(filtered, strict=False)

print("Loaded:", len(filtered), "/", len(target))
print("Missing:", len(missing))
print("Unexpected:", len(unexpected))
print("Missing examples:", missing[:20])
print("Unexpected examples:", unexpected[:20])


# Template lightning ckpt under streamingclam
ckpt = torch.load("PATH HERE", map_location='cpu') # your provided ckpt
ckpt["state_dict"] = lit.state_dict()  # after loading filtered weights into the model

# optionally reset these if you want training to start "fresh"
ckpt["epoch"] = 0
ckpt["global_step"] = 0
ckpt["optimizer_states"] = []      # so Lightning won't try to restore mismatched optimizer state
ckpt["lr_schedulers"] = []

torch.save(ckpt, "converted_lightning.ckpt")


