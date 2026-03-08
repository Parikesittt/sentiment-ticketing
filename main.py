from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/w11wo/indonesian-roberta-base-sentiment-classifier"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class RequestBody(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(body: RequestBody):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": body.text}
    )

    result = response.json()

    return result