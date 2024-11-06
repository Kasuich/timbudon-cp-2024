import asyncio
import logging
from io import BytesIO
from pprint import pformat
from time import perf_counter

import grpc
import uvicorn
from fastapi import FastAPI, File, UploadFile
from inference_pb2 import InferenceReply, InferenceRequest
from inference_pb2_grpc import InferenceServerStub
from PIL import Image

app = FastAPI()

logging.basicConfig(level=logging.INFO)


async def send_grpc_request(image_bytes):
    async with grpc.aio.insecure_channel("[::]:50052") as channel:
        stub = InferenceServerStub(channel)
        start = perf_counter()

        res: InferenceReply = await stub.inference(
            InferenceRequest(image=[image_bytes, image_bytes])
        )
        logging.info(
            f"pred = {pformat(res.pred)} in {(perf_counter() - start) * 1000:.2f}ms"
        )
        return str(res.pred)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    prediction = await send_grpc_request(image_bytes)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
