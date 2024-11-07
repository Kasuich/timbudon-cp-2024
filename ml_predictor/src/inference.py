from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34

model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()
preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@torch.no_grad()
def inference(images: List[Image.Image]) -> List[str]:
    batch = torch.stack([preprocess(image) for image in images])
    logits = model(batch)
    preds = logits.argmax(dim=1).tolist()
    str_preds = [str(pred) for pred in preds]
    return str_preds

    # TODO: impelement inference