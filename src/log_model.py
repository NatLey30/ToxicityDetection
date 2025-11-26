import argparse

import torch
from torch.utils.data import DataLoader

import mlflow
import yaml

import sys
sys.path.append(os.getcwd())

from src.utils import load_model
from src.data import load_and_prepare_datasets, set_global_seed
from src.train_functions import test_step


def parse_args():
    parser = argparse.ArgumentParser(description="Train toxicity classifier (DistilBERT)")

    # Model and data
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="models/distilbert_toxic")
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":

    import dagshub
    dagshub.init(repo_owner='NatLey30', repo_name='ToxicityDetection', mlflow=True)
    # tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    # mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri("https://dagshub.com/NatLey30/ToxicityDetection.mlflow")
    mlflow.set_experiment("toxicity-classifier")

    args = parse_args()

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["eval"]

    # Override argparse values with params from YAML
    # args.model_name = params.get("model_name", args.model_name)
    # args.max_length = params.get("max_length", args.max_length)
    # args.val_size = params.get("val_size", args.val_size)
    # args.test_size = params.get("test_size", args.test_size)
    # args.batch_size = params.get("batch_size", args.batch_size)
    # args.num_epochs = params.get("num_epochs", args.num_epochs)
    # args.learning_rate = params.get("learning_rate", args.learning_rate)
    # args.weight_decay = params.get("weight_decay", args.weight_decay)
    # args.warmup_ratio = params.get("warmup_ratio", args.warmup_ratio)
    # args.seed = params.get("seed", args.seed)
    # args.output_dir = params.get("output_dir", args.output_dir)
    # args.num_workers = params.get("num_workers", args.num_workers)

    #  Set all random seeds for reproducibility
    set_global_seed(args.seed)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    id2label = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    model, tokenizer = load_model(args.output_dir, device)

    # Load dataset
    datasets_tokenized, _ = load_and_prepare_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    test_loader = DataLoader(
        datasets_tokenized["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    with mlflow.start_run():
        mlflow.log_param("model_name", "distilbert_toxic")

        _, _, test_metrics = test_step(model, test_loader, device)

        # with open("mlflow_metrics.json", "w") as f:
        #     json.dump(test_metrics, f)

        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(args.output_dir, artifact_path="tokenizer")
