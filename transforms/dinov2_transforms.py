import torchvision.transforms as transforms
from PIL import Image
from .base import RandomGaussianBlur

def ensure_three_channels(img):
    """
    Ensure the image has three channels. If it has only one channel,
    repeat it three times.
    """
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    return img

def DINOTransform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(255, 701), interpolation=Image.BICUBIC, antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(p=1.0),
        transforms.Lambda(ensure_three_channels),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def DINOTransformPrime():
    transform_prime = transforms.Compose([
        transforms.RandomResizedCrop(
            (255, 701), interpolation=Image.BICUBIC, antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(p=0.1),
        transforms.Lambda(ensure_three_channels),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform_prime
