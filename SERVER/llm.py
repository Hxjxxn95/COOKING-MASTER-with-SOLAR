from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain Ollama model
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")
