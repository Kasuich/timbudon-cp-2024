import abc
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def predict(self, images: list[Image.Image]) -> list[str]:
        pass
