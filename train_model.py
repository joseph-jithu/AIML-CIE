"""Train a small standardized KNN crop classifier from Crop_recommendation.csv."""
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Crop_recommendation.csv"
MODEL = ROOT / "model.json"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_rows():
    rows = []
    with DATA.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for line in reader:
            values = [float(line[k]) for k in FEATURES]
            rows.append((values, line["label"].strip()))
    if not rows:
        raise ValueError("Dataset has no rows")
    return rows


def fit(rows, k=7):
    xs = [x for x, _ in rows]
    means = [sum(x[i] for x in xs) / len(xs) for i in range(len(FEATURES))]
    stds = []
    for i, mean in enumerate(means):
        variance = sum((x[i] - mean) ** 2 for x in xs) / len(xs)
        stds.append(math.sqrt(variance) or 1.0)
    points = [[(v - means[i]) / stds[i] for i, v in enumerate(x)] for x, _ in rows]
    return {"features": FEATURES, "means": means, "stds": stds, "points": points,
            "labels": [label for _, label in rows], "k": k, "dataset_rows": len(rows),
            "model": "Standardized K-Nearest Neighbors", "version": 1}


def predict(model, values, top_n=3):
    q = [(v - model["means"][i]) / model["stds"][i] for i, v in enumerate(values)]
    distances = []
    for point, label in zip(model["points"], model["labels"]):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(q, point)))
        distances.append((d, label))
    nearest = sorted(distances)[:model["k"]]
    votes = Counter()
    for dist, label in nearest:
        votes[label] += 1 / max(dist, 0.05)
    ranked = votes.most_common(top_n)
    total = sum(votes.values())
    return [(label, score / total) for label, score in ranked]


def main():
    rows = load_rows()
    indices = list(range(len(rows)))
    random.Random(42).shuffle(indices)
    cut = int(len(indices) * 0.8)
    train_rows = [rows[i] for i in indices[:cut]]
    test_rows = [rows[i] for i in indices[cut:]]
    validation_model = fit(train_rows)
    correct = sum(predict(validation_model, x, 1)[0][0] == label for x, label in test_rows)
    accuracy = correct / len(test_rows)
    model = fit(rows)
    model["holdout_accuracy"] = accuracy
    MODEL.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"Trained {model['model']} on {len(rows)} rows, {len(set(model['labels']))} crop classes")
    print(f"Stratified-style random 80/20 holdout accuracy: {accuracy:.1%} ({correct}/{len(test_rows)})")
    print(f"Saved {MODEL.name}")


if __name__ == "__main__":
    main()
