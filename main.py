from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier"
)

class RequestBody(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(body: RequestBody):
    result = classifier(body.text)

    return {
        "sentiment": result[0]["label"],
        "score": result[0]["score"]
    }