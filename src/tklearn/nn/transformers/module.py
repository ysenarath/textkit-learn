from torch import nn


def freeze(
    module: nn.Module,
    freeze_embeddings: bool = False,
    num_lower_encoder_layers: int = -1,
):
    for name, param in module.named_parameters():
        if name.startswith("embeddings.") and freeze_embeddings:
            param.requires_grad = False
        if name.startswith("encoder.layer."):
            encoder_layer_id = int(name.split(".")[2])
            if encoder_layer_id <= num_lower_encoder_layers:
                param.requires_grad = False
