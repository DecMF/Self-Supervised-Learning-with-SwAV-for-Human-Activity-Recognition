import torch
from .base import RandomGaussianBlur
import torchvision.transforms as transforms
from PIL import Image


class BarlowTwinsTransforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(255, 701), interpolation=Image.BICUBIC, antialias=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=1.0),
            # transforms.RandomSolarize(threshold=128,p=0.0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(
                (255, 701), interpolation=Image.BICUBIC, antialias=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=0.1),
            # transforms.RandomSolarize(threshold=128,p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return x, y1, y2
