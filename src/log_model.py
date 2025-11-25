import torch
import mlflow
from src.utils import load_model

import dagshub
dagshub.init(repo_owner='NatLey30', repo_name='ToxicityDetection', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/NatLey30/ToxicityDetection.mlflow")
mlflow.set_experiment("toxicity-classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = load_model("models/distilbert_toxic", device)

with mlflow.start_run():
    mlflow.log_param("model_name", "distilbert_toxic")
    mlflow.pytorch.log_model(model, "model")
