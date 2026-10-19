"""Fine-tune a shared Transformer on titles and subfield names + descriptions.

Text2Topic-style CLS pooling and [topic, title, |topic-title|, topic*title]
features, followed by a two-layer binary classifier. Every title is paired
with every taxonomy name and description. Unlisted labels are treated as negative, matching
the existing dataset pipeline. No abstract, keyword, or field name is used.

Train on all rows in train_csv_path; test_csv_path is evaluated after every epoch.

Example: python text2topic_model.py --batch_size 16 --epochs 6
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import (
    PapersDataset, add_multihot_labels, build_label_mapping, load_papers,
    normalize_text, set_seed,
)


class Text2TopicModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, tokens):
        return self.encoder(**tokens).last_hidden_state[:, 0, :]

    def score_pairs(self, title_vectors, topic_vectors):
        batch, count = title_vectors.shape[0], topic_vectors.shape[0]
        topic = topic_vectors.unsqueeze(0).expand(batch, -1, -1)
        title = title_vectors.unsqueeze(1).expand(-1, count, -1)
        features = torch.cat((topic, title, (topic - title).abs(), topic * title), dim=-1)
        return self.classifier(features).squeeze(-1)

    def forward(self, title_tokens, topic_tokens):
        # Recompute topic embeddings each training step so encoder gradients flow.
        return self.score_pairs(self.encode(title_tokens), self.encode(topic_tokens))


def load_data(args):
    train = load_papers(args.train_csv_path)
    taxonomy = pd.read_csv(args.taxonomy_path)
    if not {"subfield_id", "subfield_name", "description"}.issubset(taxonomy.columns):
        raise ValueError("Taxonomy requires subfield_id, subfield_name, and description.")
    taxonomy["subfield_id"] = pd.to_numeric(taxonomy["subfield_id"], errors="raise").astype(int)
    if taxonomy["subfield_id"].duplicated().any():
        raise ValueError("Taxonomy must contain exactly one name per subfield_id.")
    if taxonomy["subfield_name"].isna().any():
        raise ValueError("Taxonomy contains missing subfield names.")
    names = {int(row.subfield_id): normalize_text(row.subfield_name)
             for row in taxonomy.itertuples()}
    if not names or not all(names.values()):
        raise ValueError("Taxonomy must contain nonempty subfield names.")
    if taxonomy["description"].isna().any():
        raise ValueError("Taxonomy contains missing descriptions.")
    descriptions = {int(row.subfield_id): normalize_text(row.description)
                    for row in taxonomy.itertuples()}
    if not all(descriptions.values()):
        raise ValueError("Taxonomy must contain nonempty descriptions for every subfield.")
    label2id = {label: index for index, label in enumerate(sorted(names))}
    observed, _ = build_label_mapping(train)
    if set(observed) - set(names):
        raise ValueError("Some training labels have no name in the taxonomy.")
    if train.empty:
        raise ValueError("Training data must be nonempty.")
    return add_multihot_labels(train, label2id), label2id, names, descriptions


def load_inference_data(path):
    frame = pd.read_csv(path)
    if "title" not in frame.columns:
        raise ValueError("Inference CSV requires a title column; labels are optional.")
    if frame.empty:
        raise ValueError("Inference data must be nonempty.")
    return frame


def move_batch(batch, device):
    labels = batch["labels"].to(device)
    tokens = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    return tokens, labels


@torch.no_grad()
def predict(model, tokenizer, frame, topic_tokens, label_ids, topic_names, args, device):
    """Preserve input row order; encode only titles, regardless of other columns."""
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        topic_vectors = model.encode(topic_tokens)
    titles = frame["title"].fillna("").astype(str).map(normalize_text).tolist()
    probabilities = []
    for start in tqdm(
        range(0, len(titles), args.batch_size), desc="Test", unit="batch",
        dynamic_ncols=True,
    ):
        tokens = tokenizer(
            titles[start:start + args.batch_size], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model.score_pairs(model.encode(tokens), topic_vectors)
        scores = logits.float().sigmoid()
        probabilities.extend(scores.cpu().tolist())
    result = frame.copy()
    result["predicted_subfield_ids"] = [
        json.dumps([label for label, score in zip(label_ids, row) if score >= args.threshold])
        for row in probabilities
    ]
    result["predicted_subfield_names"] = [
        json.dumps([name for name, score in zip(topic_names, row) if score >= args.threshold])
        for row in probabilities
    ]
    for index, label in enumerate(label_ids):
        result[f"probability_{label}"] = [row[index] for row in probabilities]
    return result


def save_epoch_predictions(path, frame, truth, label_ids):
    """Save test rows with original label IDs and thresholded predictions."""
    if len(frame) != len(truth) or truth.shape[1] != len(label_ids):
        raise ValueError("Test rows, targets, and label IDs must be aligned.")
    sample_ids = frame["id_sample"].tolist() if "id_sample" in frame else list(range(len(frame)))
    pd.DataFrame({
        "id_sample": sample_ids,
        "ground_truth": [
            json.dumps([int(label_ids[index]) for index in np.flatnonzero(row)])
            for row in truth
        ],
        "prediction": frame["predicted_subfield_ids"].tolist(),
    }).to_csv(path, index=False)


def compute_test_metrics(truth, scores, threshold):
    """Compute metrics without printing examples or changing model weights."""
    predicted = scores >= threshold
    supported = truth.sum(axis=0) > 0
    metrics = {
        "subset_accuracy": float(accuracy_score(truth, predicted)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_mAP": float(average_precision_score(truth, scores, average="micro")),
        "macro_mAP": float(average_precision_score(
            truth[:, supported], scores[:, supported], average="macro")),
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv_path", default="papers_train.csv")
    parser.add_argument("--test_csv_path", default="papers_test.csv")
    parser.add_argument("--taxonomy_path", default="taxonomy.csv")
    parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output_dir", default="outputs/text2topic")
    parser.add_argument("--prediction_output_dir", default="outputs",
                        help="Parent directory for timestamped per-epoch prediction CSVs.")
    parser.add_argument("--log_dir", default="logs", help="Directory for training logs.")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--topic_max_length", type=int, default=128,
                        help="Maximum tokens for each subfield name + description.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.max_length, args.topic_max_length) < 1:
        parser.error("Batch size, epochs, and sequence lengths must be positive.")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1.")
    if not 0 <= args.warmup_ratio <= 1 or not 0 <= args.dropout < 1:
        parser.error("Invalid warmup_ratio or dropout.")
    if args.lr <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        parser.error("Invalid learning rate, weight decay, or worker count.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 training requires an available CUDA GPU.")
    args.precision = "fp16"
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"text2topic_{started_at}.txt"
    prediction_dir = Path(args.prediction_output_dir) / started_at
    prediction_dir.mkdir(parents=True, exist_ok=False)
    log_path.write_text(
        "Text2Topic training\n" + json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training log: {log_path}", flush=True)
    print(f"Epoch predictions: {prediction_dir}", flush=True)
    set_seed(args.seed)
    train, label2id, names, descriptions = load_data(args)
    inference_frame = load_papers(args.test_csv_path)
    if inference_frame.empty:
        raise ValueError("Test data must be nonempty.")
    observed, _ = build_label_mapping(inference_frame)
    if set(observed) - set(label2id):
        raise ValueError("Some test labels are outside the model taxonomy.")
    test_targets = add_multihot_labels(inference_frame, label2id)
    test_truth = np.stack(test_targets["labels"].to_numpy()).astype(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    topic_names = [names[label] for label in label2id]
    topic_descriptions = [descriptions[label] for label in label2id]
    topic_texts = [f"{name}: {description}"
                   for name, description in zip(topic_names, topic_descriptions)]
    topic_tokens = {key: value.to(device) for key, value in tokenizer(
        topic_texts, padding=True, truncation=True, max_length=args.topic_max_length,
        return_tensors="pt",
    ).items()}
    train_loader = DataLoader(
        PapersDataset(train, tokenizer, args.max_length), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = Text2TopicModel(AutoModel.from_pretrained(args.model_name), args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * args.warmup_ratio), num_training_steps=steps,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader, desc=f"Train epoch {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True,
        )
        for batch in progress:
            tokens, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(tokens, topic_tokens), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            total_loss += loss.item() * labels.shape[0]
            progress.set_postfix(train_loss=f"{loss.item():.4f}")
        predictions = predict(
            model, tokenizer, inference_frame, topic_tokens, list(label2id), topic_names, args, device,
        )
        save_epoch_predictions(
            prediction_dir / f"epoch{epoch}.csv", predictions, test_truth, list(label2id),
        )
        scores = predictions[[f"probability_{label}" for label in label2id]].to_numpy()
        metrics = compute_test_metrics(test_truth, scores, args.threshold)
        record = {"epoch": epoch, "train_loss": total_loss / len(train),
                  **{f"test_{key}": value for key, value in metrics.items()}}
        history.append(record)
        message = f"Epoch {epoch:02d} | " + " | ".join(
            f"{key}={value:.4f}" for key, value in record.items() if key != "epoch"
        )
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
        (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Save the final epoch; no validation-based checkpoint selection.
    model.half()
    model.encoder.save_pretrained(output / "encoder")
    tokenizer.save_pretrained(output / "encoder")
    torch.save(model.classifier.state_dict(), output / "classifier.pt")
    metadata = {
        "args": vars(args), "label_ids": list(label2id), "topic_names": topic_names,
        "topic_descriptions": topic_descriptions, "topic_texts": topic_texts,
        "epoch": args.epochs, "training_rows": len(train),
        "checkpoint_selection": "final_epoch",
        "pair_features": ["topic", "title", "abs(topic-title)", "topic*title"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    prediction_path = output / "test_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Training complete. Model and predictions saved to {output}\n")


if __name__ == "__main__":
    main()
