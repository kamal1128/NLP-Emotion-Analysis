"""scripts/regenerate_models.py
Train a simple model from data/train.txt and save model, vectorizer, and label_map to models/.

This overwrites models/emotion_model.pkl, models/vectorizer.pkl, and models/label_map.json.
"""
from pathlib import Path
import json
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train.txt"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" not in line:
                continue
            t, lbl = line.rsplit(";", 1)
            texts.append(t)
            labels.append(lbl)
    return texts, labels


def main():
    print("Loading data from", DATA)
    texts, labels = load_data(DATA)
    print("Examples:", len(texts))

    # vectorizer (use defaults so we don't accidentally filter everything)
    vect = CountVectorizer()
    X = vect.fit_transform(texts)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # save artifacts in binary mode
    model_path = MODELS / "emotion_model.pkl"
    vect_path = MODELS / "vectorizer.pkl"
    label_path = MODELS / "label_map.json"

    with model_path.open("wb") as f:
        pickle.dump(clf, f)
    with vect_path.open("wb") as f:
        pickle.dump(vect, f)

    # label map as {str(idx): label}
    label_map = {int(k): v for k, v in enumerate(le.classes_)}
    with label_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" ", model_path)
    print(" ", vect_path)
    print(" ", label_path)


if __name__ == '__main__':
    main()
