import os
import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.vectorstores.faiss import FAISS
from langserve import RemoteRunnable
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from PIL import Image
from operator import itemgetter

from RAG import embed_file, format_docs
from Prompt import RAG_PROMPT, General_PROMPT
from UploadImage import get_image
from History import print_history, add_history


LANGSERVE_ENDPOINT = "http://localhost:8000/llm/"

# 필수 디렉토리 생성 
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")


st.set_page_config(page_title="COOKING MASTER with llama3", page_icon="💬")
st.title("COOKING MASTER with llama3")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요? 사진을 올리시면 음식을 추천해드립니다!"),
    ]

with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf", "txt", "docx"],
    )
    
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(human_prefix="user", ai_prefix="ai")

# 메모리 객체 참조
memory = st.session_state.conversation_memory
conversation = ConversationChain(memory=memory, llm = RemoteRunnable(LANGSERVE_ENDPOINT))

if file:
    retriever = embed_file(file)

with st.sidebar:
    uploaded_image = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], key="image")

if uploaded_image is not None:
    get_image(uploaded_image, conversation)

print_history()


if user_input := st.chat_input():
    add_history("user", user_input)
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        
        chat_container = st.empty()
        
        if file is not None:
            prompt = RAG_PROMPT
            # 체인을 생성합니다.
            rag_chain = (
                {
                    "context": itemgetter("input")|retriever|format_docs,
                    "history": itemgetter("history"),
                    "input" : itemgetter("input")
                }
                | prompt
                | conversation.llm
                | StrOutputParser()
            )
            # 문서에 대한 질의를 입력하고, 답변을 출력합니다.
            answer = rag_chain.stream({
                    "history": str(memory.load_memory_variables({})['history']), \
                    "input": str(user_input)})
            # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            conversation.memory.save_context(inputs={"user": user_input}, outputs={"ai": "".join(chunks)})
            add_history("ai", "".join(chunks))
             
        else:
            prompt = General_PROMPT
            # 체인을 생성합니다.
            chain = prompt | conversation.llm | StrOutputParser()
            answer = chain.stream({"history": memory.load_memory_variables({})['history'], "input": user_input})  # 문서에 대한 질의
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
                chat_container.markdown("".join(chunks))
            conversation.memory.save_context(inputs={"user": user_input}, outputs={"ai": "".join(chunks)})
            add_history("ai", "".join(chunks))
