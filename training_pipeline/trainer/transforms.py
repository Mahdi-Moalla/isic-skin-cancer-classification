"""
torchvision transforms
"""

#  pylint: disable=import-error
import torch
from torchvision.transforms import v2 as torchvision_v2

train_transforms = torchvision_v2.Compose(
    [
        # torchvision_v2.PILToTensor(),
        # torchvision_v2.Resize(size=(256,256),
        #   interpolation = torchvision_v2.InterpolationMode.BICUBIC),
        # torchvision_v2.CenterCrop(size=(224, 224)),
        torchvision_v2.RandomHorizontalFlip(p=0.5),
        torchvision_v2.RandomVerticalFlip(p=0.5),
        # torchvision_v2.RandomAffine(degrees=45, translate=(0.1,0.3), scale=(0.75,1.0)),
        # torchvision_v2.ColorJitter(brightness=.5, hue=.3),
        # torchvision_v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        # torchvision_v2.RandomInvert(),
        # torchvision_v2.RandomPosterize(bits=2),
        # torchvision_v2.RandomAdjustSharpness(sharpness_factor=2),
        # torchvision_v2.RandAugment(),
        torchvision_v2.AutoAugment(torchvision_v2.AutoAugmentPolicy.IMAGENET),
        # torchvision_v2.RandomEqualize(),
        torchvision_v2.AugMix(),
        torchvision_v2.ToDtype(torch.float32, scale=True),
        torchvision_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = torchvision_v2.Compose(
    [
        # torchvision_v2.PILToTensor(),
        # torchvision_v2.Resize(size=(256,256),
        # interpolation = torchvision_v2.InterpolationMode.BICUBIC ),
        # torchvision_v2.CenterCrop(size=(224, 224)),
        torchvision_v2.ToDtype(torch.float32, scale=True),
        torchvision_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
