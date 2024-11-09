import asyncio
import logging
from concurrent import futures
from io import BytesIO
from time import perf_counter

import grpc
from model.base_model import BaseModel, PredictResult
from model.ocr_pipeline import OcrPipeline
from pb.inference_pb2 import ImageRequest, ImageResponse
from pb.inference_pb2_grpc import (
    ImageRecognitionServiceServicer,
    add_ImageRecognitionServiceServicer_to_server,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)


class InferenceService(ImageRecognitionServiceServicer):

    def __init__(self, model: BaseModel):
        self._model = model
        super().__init__()

    def open_image(self, image: bytes) -> Image.Image:
        image = Image.open(BytesIO(image))
        return image

    async def RecognizeImage(self, request: ImageRequest, context) -> ImageResponse:

        logging.info(f"Received request")
        start = perf_counter()
        image = self.open_image(request.image_data)
        # TODO: Adjust parameters where all features of model will be ready
        preds: PredictResult = self._model.predict(image, False, 0)
        logging.info(f"Done in {(perf_counter() - start) * 1000:.2f}ms")

        response = ImageResponse()
        response.recognized_text = preds.raw_text
        response.marked_image = preds.pred_img
        response.attribute_1 = preds.attribute1 if preds.attribute1 else "" 
        response.attribute_2 = preds.attribute2 if preds.attribute2 else ""
        response.attribute_3 = preds.attribute3 if preds.attribute3 else ""
        return response


async def serve():
    model = OcrPipeline()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ImageRecognitionServiceServicer_to_server(InferenceService(model=model), server)
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f" Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
