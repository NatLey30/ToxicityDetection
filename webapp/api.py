from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.utils import load_model
from src.prediction import predict_with_scores

app = FastAPI(title="Toxicity Detection API")

# —— Load model once at startup ——
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model("models/best_model", device=device)

# These are your labels
id2label = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

app = FastAPI(title="Toxicity API")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input_data: TextInput):
    result = predict_with_scores(
        model=model,
        tokenizer=tokenizer,
        text=input_data.text,
        id2label=id2label,
        device=device,
        threshold=0.5,
        top_k=3
    )
    return result
