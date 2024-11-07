import torch
import torchvision.transforms as T
from model.base_model import BaseModel
from torchvision.models import ResNet34_Weights, resnet34


class DummyModel(BaseModel):

    def __init__(self) -> None:
        self._model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()

        self._preprocess = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, images):
        batch = torch.stack([self._preprocess(image) for image in images])
        logits = self._model(batch)
        preds = logits.argmax(dim=1).tolist()
        str_preds = [str(pred) for pred in preds]
        return str_preds
