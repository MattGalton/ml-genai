import torchvision
from torchvision import transforms
from PIL import Image

from mattg import HOME


class Binarize:
    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.convert("L").point(lambda x: 0 if x < self.threshold else 255)


class BMNIST(torchvision.datasets.MNIST):
    """Binarized MNIST dataset"""
    def __init__(self, train=True):
        transform = transforms.Compose([
            Binarize(128),
            transforms.ToTensor(),
        ])

        super().__init__(root=HOME, train=train, download=True, transform=transform)