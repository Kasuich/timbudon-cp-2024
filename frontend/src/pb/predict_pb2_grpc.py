# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import pb.predict_pb2 as predict__pb2

GRPC_GENERATED_VERSION = '1.67.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in predict_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ImageRecognitionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RecognizeImage = channel.unary_unary(
                '/image_recognition.ImageRecognitionService/RecognizeImage',
                request_serializer=predict__pb2.ImageRequest.SerializeToString,
                response_deserializer=predict__pb2.ImageResponse.FromString,
                _registered_method=True)


class ImageRecognitionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RecognizeImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        
        # response = ImageRecognitionServiceServicer_pb2.ImageResponse()
        # response.recognized_text = "Пример распознанного текста"
        # response.marked_image = b'...'  # Здесь вы можете добавить байты изображения
        # response.attribute_1 = "Атрибут 1"
        # response.attribute_2 = "Атрибут 2"
        # response.attribute_3 = "Атрибут 3"

        # return response


def add_ImageRecognitionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RecognizeImage': grpc.unary_unary_rpc_method_handler(
                    servicer.RecognizeImage,
                    request_deserializer=predict__pb2.ImageRequest.FromString,
                    response_serializer=predict__pb2.ImageResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'image_recognition.ImageRecognitionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('image_recognition.ImageRecognitionService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ImageRecognitionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RecognizeImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/image_recognition.ImageRecognitionService/RecognizeImage',
            predict__pb2.ImageRequest.SerializeToString,
            predict__pb2.ImageResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
