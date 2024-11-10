import streamlit as st
from PIL import Image
import io
import base64

import pb.predict_pb2 as predict_pb2

def display_image_response(image_response: predict_pb2.ImageResponse):
    st.markdown(
    """
    <h2 style="font-size: calc(24px + 2vw); margin: 10px 0;">
        Результаты:
    </h2>
    """,
    unsafe_allow_html=True
    )

    # Отображаем распознанный текст
    st.markdown(
    """
    <h3 style="font-size: calc(20px + 2vw); margin-top: 10px; padding: 0">
        Распознанный текст
    </h3>
    """,
    unsafe_allow_html=True
    )
    styled_text = f"""
    <div style="background-color: #f0f0f5; padding: 10px; border-radius: 5px; font-size: 16px; color: #333;">
        <pre>{image_response.recognized_text}</pre>
    </div>
    """
    st.markdown(styled_text, unsafe_allow_html=True)

    # Отображаем изображение
    st.markdown(
    """
    <h3 style="font-size: calc(20px + 2vw); margin-top: 30px;">
        Размеченное изображение
    </h3>
    """,
    unsafe_allow_html=True
    )
    # Загружаем изображение
    buffered = io.BytesIO(image_response.marked_image)
    # image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Добавляем стиль для изображения
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{}" style="border: 2px solid #4CAF50; border-radius: 10px; width: 600px; height: auto;"/>
        </div>
        """.format(img_str),  # Предполагается, что у вас есть base64 строка изображения
        unsafe_allow_html=True
    )

    # Отображаем изображение с подписью
    # st.image(image, caption='Размеченное изображение', use_container_width=True)

    # Отображаем дополнительные атрибуты
    st.subheader("Атрибуты")
    st.write("Атрибут 1:", image_response.attribute_1)
    st.write("Атрибут 2:", image_response.attribute_2)
    st.write("Атрибут 3:", image_response.attribute_3)
