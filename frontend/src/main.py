import streamlit as st
from PIL import Image
import io

from utils.send import run_grpc_client
from utils.results import display_image_response
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="font-size: calc(28px + 2vw); text-align: center; margin: 10px 0;">
        Распознавание маркировки
    </h1>
    """,
    unsafe_allow_html=True
)
option = st.selectbox("Выберите действие:", ("Загрузить изображение", "Сделать фотографию"))

if option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    enable = st.checkbox("Делать поиск по excel")
    if uploaded_file is not None:
        with Image.open(uploaded_file) as img:
            byte_io = io.BytesIO()
            img.save(byte_io, format='JPEG')
            image_bytes = byte_io.getvalue()

        responce = run_grpc_client(
            image=image_bytes,
            search_flag=enable,
        )
        
        display_image_response(responce)

elif option == "Сделать фотографию":
    picture = st.camera_input("Сделайте фотографию")
    enable = st.checkbox("Делать посик по exel")
    if picture is not None:
        bytes_data = picture.getvalue()

        responce = run_grpc_client(
            image=bytes_data,
            search_flag=enable,
        )
        display_image_response(responce)


    