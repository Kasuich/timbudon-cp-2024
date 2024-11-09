import grpc
import pb.predict_pb2 as predict_pb2
import pb.predict_pb2_grpc as predict_pb2_grpc


def run_grpc_client(image: bytes, search_flag: bool = False):
    with grpc.insecure_channel("ml_predictor:50052") as channel:
        stub = predict_pb2_grpc.ImageRecognitionServiceStub(channel)
        response = stub.RecognizeImage(
            predict_pb2.ImageRequest(
                image_data=image,
                search_flag=search_flag,
            )
        )

        return response
