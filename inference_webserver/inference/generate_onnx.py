"""
generate onnx model
"""

# pylint: disable=import-error
import json

import fire
import torch
from addict import Dict

config={}
with open("trainer_config.json", "r", encoding="utf-8") as f:
    config=Dict(json.load(f))
# from config import config


def generate_onnx(img_shape=(3, 224, 224), output_file='./model.onnx'):
    """
    generate onnx model from  pytorch model
    """
    model = torch.load('model.pth', weights_only=False)

    input_img = torch.zeros(1, *img_shape, dtype=torch.float)

    input_tab_feats = torch.zeros(1,
            len(config.tab_features),
            dtype=torch.float)

    inputs = (input_img, input_tab_feats)

    model.to_onnx(output_file, inputs, export_params=True)


if __name__ == '__main__':
    fire.Fire(generate_onnx)
