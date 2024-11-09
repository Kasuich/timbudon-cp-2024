from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ImageRequest(_message.Message):
    __slots__ = ("image_data", "search_flag")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    SEARCH_FLAG_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    search_flag: bool
    def __init__(self, image_data: _Optional[bytes] = ..., search_flag: bool = ...) -> None: ...

class ImageResponse(_message.Message):
    __slots__ = ("recognized_text", "marked_image", "attribute_1", "attribute_2", "attribute_3")
    RECOGNIZED_TEXT_FIELD_NUMBER: _ClassVar[int]
    MARKED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_1_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_2_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_3_FIELD_NUMBER: _ClassVar[int]
    recognized_text: str
    marked_image: bytes
    attribute_1: str
    attribute_2: str
    attribute_3: str
    def __init__(self, recognized_text: _Optional[str] = ..., marked_image: _Optional[bytes] = ..., attribute_1: _Optional[str] = ..., attribute_2: _Optional[str] = ..., attribute_3: _Optional[str] = ...) -> None: ...
