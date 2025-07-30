"""
pytest module
testing transfer learning modifications in isic_model.py
"""

import torchvision
from torch import nn

from training_pipeline.trainer.isic_model import get_toplast_layer


def get_last_layer(model_name, transfer_learning_layer):
    """
    test helper function
    get last layer
    """

    model = torchvision.models.get_model(model_name)

    toplast_layer = get_toplast_layer(model, transfer_learning_layer[:-1])

    last_layer_id = transfer_learning_layer[-1]
    if isinstance(last_layer_id, str):
        return getattr(toplast_layer, last_layer_id)
    if isinstance(last_layer_id, int):
        return toplast_layer[last_layer_id]
    raise TypeError(f"type of last layer id ({type(last_layer_id)}) is unsupported")


def test_isic_model_get_toplast_layer():
    """
    testing get_toplast_layer
    """

    last_layer = get_last_layer("efficientnet_b0", ['classifier', 1])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 1280

    last_layer = get_last_layer("efficientnet_b7", ['classifier', 1])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 2560

    last_layer = get_last_layer("efficientnet_v2_s", ['classifier', 1])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 1280

    last_layer = get_last_layer("regnet_x_32gf", ['fc'])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 2520

    last_layer = get_last_layer("vit_b_16", ['heads', 'head'])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 768

    last_layer = get_last_layer("maxvit_t", ['classifier', 5])

    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 512
