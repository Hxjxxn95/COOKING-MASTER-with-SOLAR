import requests
import streamlit as st
from History import add_history
from translate import Translator

import base64
from PIL import Image
import io

def translate_to_korean(text):
    translator = Translator(to_lang="ko")
    translation = translator.translate(text)
    return translation

@st.cache_resource(show_spinner="uploading...")
def get_image(uploaded_image, _conversation):
    response = requests.post(
    "http://localhost:8000/YOLO",
    files={"file": (uploaded_image.name, uploaded_image, "image/jpeg")}
    )
    if response.status_code != 200:
        st.write("에러 발생:", response.text)
    else:
        response_text = response.json()["result"]
        image = response.json()["image"]
        object_ = [string for string in response_text]
        object_ = ", ".join(object_)
        object_korean = translate_to_korean(object_)
        if len(object_) == 0:
            chunks = ["재료를 인식하지 못하였습니다. 다른 이미지를 시도해주세요."]
        else:
            chunks = [f"음식을 추천하기 위해서 찾은 음식들은 {object_korean}입니다! 어떤 음식을 찾아드릴까요?"]
        _conversation.memory.save_context(inputs={"user": "냉장고에서 음식 찾아줘"}, outputs={"ai": "".join(chunks)})
        add_history("ai", image)
        add_history("ai", "".join(chunks))