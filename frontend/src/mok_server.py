import grpc
from concurrent import futures
import time
import pb.predict_pb2 as predict_pb2
import pb.predict_pb2_grpc as predict_pb2_grpc

from PIL import Image
import io

class ImageRecognition(predict_pb2_grpc.ImageRecognitionServiceServicer):
    def RecognizeImage(self, request, context):

        response = predict_pb2.ImageResponse()
        response.recognized_text = "Пример распознанного текста"

        image_path = 'photo_2024-11-09_08-57-23.jpg'
        with Image.open(image_path) as img:
            byte_io = io.BytesIO()
            img.save(byte_io, format='JPEG')
            image_bytes = byte_io.getvalue()
        response.marked_image = image_bytes
        
        response.attribute_1 = "Атрибут 1"
        response.attribute_2 = "Атрибут 2"
        response.attribute_3 = "Атрибут 3"
        print(response)
        return response
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_ImageRecognitionServiceServicer_to_server(ImageRecognition(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server is running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
