import io

from model.base_model import BaseModel, PredictResult
from PIL import Image


class PlaceholderModel(BaseModel):

    def predict(images, search_in_data) -> PredictResult:

        res = PredictResult()
        with Image.open("photo_2024-11-09_18-54-13.jpg") as img:
            byte_io = io.BytesIO()
            img.save(byte_io, format="JPEG")
            image_bytes = byte_io.getvalue()
        res.pred_img = image_bytes
        res.attribute1 = "123"
        res.attribute2 = "234324324"
        res.attribute3 = "sdfsdfsdf"
        return res
