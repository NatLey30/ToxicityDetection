import streamlit as st
import pandas as pd
import torch

import sys
import os
sys.path.append(os.path.abspath("."))

from src.utils import load_model
from src.prediction import predict_with_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/distilbert_toxic"
model, tokenizer = load_model(model_path, device)

# id2label from config
id2label = list(model.config.id2label.values())

# Load TEST dataset (raw CSV)
TEST_PATH = "data/jigsaw/test.csv"
df_test = pd.read_csv(TEST_PATH)

# The column with the text
texts = df_test["comment_text"].tolist()

# Streamlit UI---
st.title("Toxic Comment Classifier – Test Set Explorer")

st.write("Selecciona un comentario del dataset de test:")

selected_text = st.selectbox("Comentario:", texts)

st.write("### Texto seleccionado:")
st.write(selected_text)

if st.button("Clasificar"):
    result = predict_with_scores(
        model=model,
        tokenizer=tokenizer,
        text=selected_text,
        id2label=id2label,
        device=device,
        threshold=0.5,
        top_k=3
    )

    st.write("### Resultados:")
    st.json(result)
