import torch
import random
import torchvision.transforms.v2 as transforms


class RandomGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            t = transforms.GaussianBlur(kernel_size=5, sigma=sigma)
            return t(img)
        else:
            return img


class RandomGaussianNoise:
    """Apply randomly, with a given probability, a white noise into a torch.Tensor image.
    """

    def __init__(self, mean=0.0, std=1.0, p=0.5, clip=(-1.0, 1.0)):
        self.mean = mean
        self.std = std
        self.p = p
        self.clip = clip

    def __repr__(self) -> str:
        return f'RandomGaussianNoise(mean={self.mean}, std={self.std}, p={self.p})'

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p < random.random():
            return img

        noise = torch.randn(img.size()) * self.std + self.mean
        return torch.clip(img+noise, min=self.clip[0], max=self.clip[1])
