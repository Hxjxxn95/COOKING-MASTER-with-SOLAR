from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
                    """
                    너는 제공된 제료로 음식을 만드는 요리사야, 레시피에 대해 설명해줘
                    Context: {context}
                    History: {history}
                    Question: {input}
                    Answer:
                    """
                    )
General_PROMPT = ChatPromptTemplate.from_template(
                """
                History: {history}
                다음의 질문에 간결하게 답변해 주세요:\n{input}
                """
            )
