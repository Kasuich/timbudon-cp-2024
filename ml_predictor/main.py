import asyncio
import logging
from concurrent import futures
from io import BytesIO
from time import perf_counter
import os
import requests
from dotenv import load_dotenv
import grpc
from model.base_model import BaseModel, PredictResult
from model.ocr_pipeline import OcrPipeline
from pb.inference_pb2 import ImageRequest, ImageResponse
from pb.inference_pb2_grpc import (
    ImageRecognitionServiceServicer,
    add_ImageRecognitionServiceServicer_to_server,
)
from PIL import Image

# Загрузка переменных из .env файла
load_dotenv()
BASE_URL = os.getenv("ONEC_BASE_URL")
USER = os.getenv("ONEC_USER")
PASSWORD = os.getenv("ONEC_PASSWORD")

logging.basicConfig(level=logging.INFO)

class InferenceService(ImageRecognitionServiceServicer):
    def __init__(self, model: BaseModel):
        self._model = model
        super().__init__()

    def open_image(self, image: bytes) -> Image.Image:
        return Image.open(BytesIO(image))

    async def RecognizeImage(self, request: ImageRequest, context) -> ImageResponse:
        logging.info("Received request")
        start = perf_counter()

        image = self.open_image(request.image_data)
        preds: PredictResult = self._model.predict(image, False, 0)

        logging.info(f"Done in {(perf_counter() - start) * 1000:.2f}ms")
        
        # Формируем ответ для клиента
        response = ImageResponse()
        response.recognized_text = preds.raw_text
        response.marked_image = preds.pred_img
        response.attribute_1 = preds.attribute1 if preds.attribute1 else ""
        response.attribute_2 = preds.attribute2 if preds.attribute2 else ""
        response.attribute_3 = preds.attribute3 if preds.attribute3 else ""

        # Отправляем данные в 1С
        self.send_data_to_1c(preds.raw_text)

        return response

    def send_data_to_1c(self, recognized_text):
        url = f"{BASE_URL}/Catalog_Example" 
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "RecognizedText": recognized_text
        }

        response = requests.post(url, json=data, headers=headers, auth=(USER, PASSWORD))
        
        if response.status_code in (200, 201):
            logging.info("Распознанный текст успешно отправлен в 1С")
        else:
            logging.error(f"Ошибка при отправке данных в 1С: {response.status_code} {response.text}")

async def serve():
    model = OcrPipeline()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ImageRecognitionServiceServicer_to_server(InferenceService(model=model), server)
    address = "[::]:50052"
    server.add_insecure_port(address)
    logging.info(f"Starting server on {address}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
