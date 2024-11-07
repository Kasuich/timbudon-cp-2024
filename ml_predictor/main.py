import asyncio
import logging
from concurrent import futures
from io import BytesIO
from time import perf_counter

import grpc
from model.base_model import BaseModel
from model.dummy_model import DummyModel
from pb.inference_pb2 import InferenceReply, InferenceRequest
from pb.inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server
from PIL import Image

logging.basicConfig(level=logging.INFO)


class InferenceService(InferenceServer):

    def __init__(self, model: BaseModel):
        self._model = model
        super().__init__()

    def open_image(self, image: bytes) -> Image.Image:
        image = Image.open(BytesIO(image))
        return image

    async def inference(self, request: InferenceRequest, context) -> InferenceReply:

        # TODO: implement inference
        logging.info(f"Received request")
        start = perf_counter()
        images = list(map(self.open_image, request.image))
        preds = self._model.predict(images)
        logging.info(f"Done in {(perf_counter() - start) * 1000:.2f}ms")
        return InferenceReply(pred=preds)


async def serve():
    model = DummyModel()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServerServicer_to_server(InferenceService(model=model), server)
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f" Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
