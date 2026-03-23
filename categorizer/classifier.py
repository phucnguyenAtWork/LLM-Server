"""
ML Categorization Classifier — TF-IDF + Logistic Regression
=============================================================
Trained on labeled synthetic data, saved to categorizer/saved/.
Used as the fallback when rule-based matching fails.
"""

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

SAVE_DIR  = Path(__file__).parent / "saved"
MODEL_PATH = SAVE_DIR / "classifier.pkl"

# ── Minimal TF-IDF implementation (no scikit-learn required) ─────────────────

class TfidfVectorizer:
    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if len(t) > 1]

    def fit(self, corpus: list[str]) -> "TfidfVectorizer":
        N = len(corpus)
        df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            for tok in set(self._tokenize(doc)):
                df[tok] += 1

        # Keep top-max_features by document frequency
        top = sorted(df.items(), key=lambda x: x[1], reverse=True)[: self.max_features]
        self.vocab = {tok: i for i, (tok, _) in enumerate(top)}
        self.idf = np.array(
            [math.log((N + 1) / (df[tok] + 1)) + 1 for tok, _ in top],
            dtype=np.float32,
        )
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        V = len(self.vocab)
        out = np.zeros((len(corpus), V), dtype=np.float32)
        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    out[i, j] = (cnt / len(tokens)) * self.idf[j]
            # L2 normalize
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)


# ── Softmax Logistic Regression (one-vs-rest via numpy) ──────────────────────

class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 200, C: float = 1.0):
        self.lr     = lr
        self.epochs = epochs
        self.C      = C          # inverse regularization strength
        self.W: np.ndarray = np.array([])
        self.b: np.ndarray = np.array([])
        self.classes: list = []

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: list[str]) -> "LogisticRegression":
        self.classes = sorted(set(y))
        K = len(self.classes)
        label_map = {c: i for i, c in enumerate(self.classes)}
        Y = np.array([label_map[c] for c in y])

        # One-hot
        Y_oh = np.zeros((len(Y), K), dtype=np.float32)
        Y_oh[np.arange(len(Y)), Y] = 1.0

        n, d = X.shape
        self.W = np.zeros((d, K), dtype=np.float32)
        self.b = np.zeros(K, dtype=np.float32)

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probs  = self._softmax(logits)
            dL     = (probs - Y_oh) / n
            self.W -= self.lr * (X.T @ dL + (1 / self.C) * self.W)
            self.b -= self.lr * dL.sum(axis=0)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> list[str]:
        idx = self.predict_proba(X).argmax(axis=1)
        return [self.classes[i] for i in idx]


# ── Wrapper ───────────────────────────────────────────────────────────────────

class TransactionClassifier:
    def __init__(self):
        self.vectorizer  = TfidfVectorizer(max_features=500)
        self.model       = LogisticRegression(lr=0.05, epochs=300, C=5.0)
        self.is_trained  = False

    def train(self, texts: list[str], labels: list[str]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (category, confidence 0-1)."""
        if not self.is_trained:
            return "General", 0.0
        X    = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0]
        idx  = prob.argmax()
        return self.model.classes[idx], float(prob[idx])

    def save(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "TransactionClassifier":
        if not MODEL_PATH.exists():
            return cls()
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
