import asyncio
import logging
from concurrent import futures
from io import BytesIO
from time import perf_counter

import grpc
from model.base_model import BaseModel, PredictResult
from model.placeholder_model import PlaceholderModel
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

    async def inference(self, request: ImageRequest, context) -> ImageResponse:

        logging.info(f"Received request")
        start = perf_counter()
        image = self.open_image(request.image_data)
        preds: PredictResult = self._model.predict(image)
        logging.info(f"Done in {(perf_counter() - start) * 1000:.2f}ms")

        response = ImageResponse()
        response.recognized_text()
        response.marked_image = preds.pred_img
        response.attribute_1 = preds.attribute1
        response.attribute_2 = preds.attribute2
        response.attribute_3 = preds.attribute3
        return response


async def serve():
    model = PlaceholderModel()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ImageRecognitionServiceServicer_to_server(InferenceService(model=model), server)
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f" Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
