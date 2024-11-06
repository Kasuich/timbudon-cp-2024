import asyncio
import logging
from io import BytesIO
from pprint import pformat
from time import perf_counter

import grpc
from inference_pb2 import InferenceReply, InferenceRequest
from inference_pb2_grpc import InferenceServerStub
from PIL import Image

image = Image.open("cat.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_bytes = buffered.getvalue()

logging.basicConfig(level=logging.INFO)


async def main():
    async with grpc.aio.insecure_channel("[::]:50052 ") as channel:
        stub = InferenceServerStub(channel)
        start = perf_counter()

        res: InferenceReply = await stub.inference(
            InferenceRequest(image=[image_bytes, image_bytes, image_bytes])
        )
        logging.info(
            f"pred = {pformat(res.pred)} in {(perf_counter() - start) * 1000:.2f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
