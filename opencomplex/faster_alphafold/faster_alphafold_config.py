import ml_collections as mlc


faster_alphafold_config = mlc.ConfigDict({
    'layer_norm': False,
    'softmax': False,
    'attention': False,
    'outer_product_mean': False,
    'triangle_multiplicative_update': False,
})
