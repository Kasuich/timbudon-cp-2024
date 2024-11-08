import streamlit as st
from PIL import Image

from utils.send import run_grpc_client

st.title("Фотографирование или загрузка изображения")

option = st.selectbox("Выберите действие:", ("Загрузить изображение", "Сделать фотографию"))
enable = st.checkbox("Делать посик по exel")

if option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_container_width=True)

elif option == "Сделать фотографию":
    picture = st.camera_input("Сделайте фотографию")
    if picture is not None:
        bytes_data = picture.getvalue()
        run_grpc_client(
            image=bytes_data,
            search_flag=enable,
        )

    