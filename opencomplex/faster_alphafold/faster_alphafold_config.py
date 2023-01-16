import torch
import ml_collections as mlc


faster_alphafold_config = mlc.ConfigDict({
    'layer_norm': True,
    'softmax': False,
    'attention': True,
    'outer_product_mean': True,
    'triangle_multiplicative_update': True,
}) if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else mlc.ConfigDict({
    'layer_norm': False,
    'softmax': False,
    'attention': False,
    'outer_product_mean': False,
    'triangle_multiplicative_update': False,
})
