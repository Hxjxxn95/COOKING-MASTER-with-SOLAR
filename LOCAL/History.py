import streamlit as st
from langchain_core.messages import ChatMessage
import base64
from PIL import Image
import io

def print_history():
    for msg in st.session_state.messages:
        if len(msg.content) > 100000:
            #decoding image
            img_bytes = base64.b64decode(msg.content)
            img = Image.open(io.BytesIO(img_bytes))
            st.chat_message(msg.role).image(img)
        else:
            st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))