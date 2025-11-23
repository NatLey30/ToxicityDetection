# ToxicityDetection

## Project Description

This project implements an end-to-end pipeline for toxic comment detection using modern Natural Language Processing (NLP) techniques and Explainable Artificial Intelligence (XAI). The objective is to train, evaluate, and analyse a text classification model capable of identifying toxic content in user-generated text, while simultaneously providing transparent explanations for its predictions.

In addition to the core machine learning component, the project follows a full machine-learning lifecycle design. It includes experiment tracking with MLflow, dataset versioning and reproducibility through DagsHub, model packaging via Docker, and the development of a simple web interface to demonstrate real-time inference and explanation outputs.

The project is designed both as a practical application of XAI methods for NLP and as an integrated MLOps case study.

## Problem Statement

The task addressed in this project is **toxicity classification**: given a text input (e.g., a comment in an online platform), the model must predict whether the content is toxic or non-toxic.

This problem is relevant in contexts such as automated content moderation, social media safety, and online community management. Explainability is a central requirement in this domain: stakeholders such as platform moderators, compliance teams, and end-users must be able to understand why a model assigns a toxicity label, particularly in borderline or sensitive cases.

For this reason, the project integrates both global and local XAI techniques to provide interpretable insights into the model’s behaviour.

## Dataset

This project uses the **Jigsaw Toxic Comment Classification Dataset**, a publicly available benchmark dataset commonly used for toxicity detection.  
It contains user comments annotated with various types of toxicity. For the purposes of this project, the task is restricted to a binary classification: *toxic* vs. *non-toxic*.

The dataset is programmatically loaded using the Hugging Face `datasets` library to ensure reproducibility and avoid manual data handling.

## PART 1. MODEL FINETUNING
### Candidate Models for Fine-Tuning

Several transformer-based language models can be used as the base architecture for fine-tuning. The candidate models considered include:

DistilBERT (`distilbert-base-uncased`)
- Approximately 66M parameters.
- Fast to train and lightweight for inference.
- Retains around 95% of the performance of BERT-base.
- Efficient for Docker deployment and real-time applications.
- Widely used as a baseline model in XAI research due to its stability and transparency.

BERT-base (`bert-base-uncased`)
- Around 110M parameters.
- Strong performance on most NLP tasks.
- Slower to train and more resource-intensive.
- Suitable for comparative analysis with compact models.

RoBERTa-base (`roberta-base`)
- High performance on many text classification benchmarks.
- Computationally heavier and slower during inference.
- More expensive for explanation methods such as SHAP and LIME.

DistilRoBERTa (`distilroberta-base`)
- Balanced trade-off between performance and efficiency.
- Less commonly used in explainability studies.

### Selected model

For this project, **DistilBERT (`distilbert-base-uncased`)** is selected. This choice is supported by the following considerations:

1. **Training efficiency**  
   DistilBERT is significantly smaller than BERT-base, allowing faster fine-tuning and more iteration cycles. This facilitates extensive experimentation, including multiple XAI evaluations.

2. **Suitability for explainability**  
   The reduced complexity of DistilBERT makes gradient-based saliency methods, LIME, and SHAP more computationally feasible. Its architecture is also widely adopted in XAI research.

3. **Ease of deployment**  
   Since the project includes a Dockerised FastAPI inference service, model size and inference speed are practical constraints. DistilBERT enables low-latency prediction and compact Docker images.

4. **Reproducibility**  
   DistilBERT is a well-documented and commonly used model, which ensures that the results can be meaningfully compared to existing literature and reproduced easily.

5. **Compatibility with XAI workflows**  
   Larger models increase the computational cost and runtime of explainability methods. DistilBERT remains efficient and stable for repeated XAI evaluations.

Although DistilBERT is the chosen model, alternative models such as BERT-base may be fine-tuned for comparison, particularly when studying the relationship between model complexity and explanation fidelity.

