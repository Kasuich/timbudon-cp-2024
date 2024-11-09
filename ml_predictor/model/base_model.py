import abc
from dataclasses import dataclass
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34


@dataclass
class PredictResult:
    raw_text: str | None = None
    # image in bytes with boxes and text on it
    pred_img: str | None = None
    # unknow data from excel, None if search_in_data is False
    attribute1: str | None = None
    attribute2: str | None = None
    attribute3: str | None = None


class BaseModel(abc.ABC):

    @abc.abstractmethod
    def predict(
        self, images: Image.Image, search_in_data: bool, dist_threshold: float
    ) -> PredictResult:
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
        PredictResult
            If search_in_data is True, returns full data from excel
            If False, return only OCR result
        """
        pass
