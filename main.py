from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import chatbot
from cv import cv as cv_assistant

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"message": "healthy"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = chatbot(request.message)
    return {"response": response}


@app.post("/cv")
async def cv(request: ChatRequest):
    response = cv_assistant(request.message)
    return {"response": response}