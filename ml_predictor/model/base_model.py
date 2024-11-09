import abc
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def predict(
        self, images: list[Image.Image], search_in_data: bool, dist_threshold: float
    ) -> list[str]:
        """Get predict from ML OCR Model

        Parameters
        ----------
        images : list[Image.Image]
            List with images to be predicted
        search_in_data : bool
            Flag, if true, get missing data from excel file
        dist_threshold : float
            Distance threshold to cut out unknown images

        Returns
        -------
        list[str]
            If search_in_data is True, returns full data from excel
            If False, return only OCR result
        """
