#python veersion
import sys
print(sys.version)

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
from chain import chain
from chat import chain as chat_chain
from translator import chain as EN_TO_KO_chain
from llm import llm as model
from YOLO import YOLO
from langchain_core.runnables import Runnable
from fastapi import FastAPI, UploadFile, File
import json


class YOLORunnable(Runnable):
    def __init__(self):
        self.yolo = YOLO()

    def run(self, input):
        return self.yolo(input)

    def invoke(self, *args, **kwargs):
        return self.run(*args, **kwargs)

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/llm")


add_routes(app, model, path="/llm")


yolo_runnable = YOLORunnable()
add_routes(app, yolo_runnable, path="/YOLO")

@app.post("/YOLO")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = yolo_runnable.run(contents)        
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
