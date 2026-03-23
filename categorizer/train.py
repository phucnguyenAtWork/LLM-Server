"""
Hybrid Categorizer — Training Script
======================================
Trains the ML classifier on synthetic labeled data and saves it.

Usage:
    python -m categorizer.train

The rule-based layer needs no training — it's always active.
The ML classifier is a fallback for descriptions that don't match any rule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from categorizer.classifier import TransactionClassifier

# ── Synthetic labeled training data ──────────────────────────────────────────
# Format: (description, category)
# Add more examples here to improve accuracy.

TRAINING_DATA = [
    # Food
    ("grab food order hcm", "Food"),
    ("shopee food delivery", "Food"),
    ("baemin order lunch", "Food"),
    ("kfc bucket meal", "Food"),
    ("mcdonalds drive thru", "Food"),
    ("highlands coffee americano", "Food"),
    ("starbucks latte", "Food"),
    ("pizza hut delivery", "Food"),
    ("restaurant dinner payment", "Food"),
    ("cafe breakfast morning", "Food"),
    ("com tam quan an", "Food"),
    ("bun bo hue", "Food"),
    ("pho beef noodle", "Food"),
    ("banh mi sandwich", "Food"),
    ("food delivery app", "Food"),
    ("lunch payment office", "Food"),
    ("dinner family restaurant", "Food"),
    ("snack convenience store food", "Food"),
    ("phuc long milk tea", "Food"),
    ("the coffee house drink", "Food"),
    ("gofood order", "Food"),
    ("lotteria burger", "Food"),
    ("jollibee chicken", "Food"),

    # Transport
    ("grab car trip", "Transport"),
    ("grab bike ride", "Transport"),
    ("be app taxi", "Transport"),
    ("taxi fare payment", "Transport"),
    ("parking fee motorbike", "Transport"),
    ("petrol station fuel", "Transport"),
    ("xang xe may", "Transport"),
    ("bus ticket payment", "Transport"),
    ("xe buyt monthly pass", "Transport"),
    ("metro card top up", "Transport"),
    ("parking lot fee", "Transport"),
    ("xanh sm car rental", "Transport"),

    # Shopping
    ("shopee online purchase", "Shopping"),
    ("lazada order payment", "Shopping"),
    ("tiki book purchase", "Shopping"),
    ("amazon international order", "Shopping"),
    ("vinmart grocery", "Shopping"),
    ("winmart supermarket", "Shopping"),
    ("circle k convenience", "Shopping"),
    ("7 eleven store", "Shopping"),
    ("guardian beauty", "Shopping"),
    ("watsons skincare", "Shopping"),
    ("fahasa bookstore", "Shopping"),
    ("mall shopping center", "Shopping"),
    ("clothing store purchase", "Shopping"),
    ("online shopping payment", "Shopping"),

    # Bills
    ("fpt internet monthly bill", "Bills"),
    ("viettel mobile bill", "Bills"),
    ("vnpt broadband payment", "Bills"),
    ("evn electricity bill", "Bills"),
    ("tien dien thang", "Bills"),
    ("water bill payment", "Bills"),
    ("tien nuoc thang", "Bills"),
    ("thanh toan hoa don internet", "Bills"),
    ("rent monthly payment", "Bills"),
    ("tien nha thue", "Bills"),
    ("insurance premium payment", "Bills"),
    ("phone bill payment", "Bills"),

    # Health
    ("pharmacity medicine", "Health"),
    ("long chau pharmacy", "Health"),
    ("hospital consultation fee", "Health"),
    ("clinic doctor visit", "Health"),
    ("medicine prescription", "Health"),
    ("nha thuoc mua thuoc", "Health"),
    ("benh vien kham benh", "Health"),
    ("health checkup payment", "Health"),
    ("dental clinic", "Health"),
    ("eye exam glasses", "Health"),

    # Education
    ("university tuition fee", "Education"),
    ("hoc phi truong dai hoc", "Education"),
    ("online course payment udemy", "Education"),
    ("coursera subscription", "Education"),
    ("english class tuition", "Education"),
    ("book purchase study", "Education"),
    ("school supplies", "Education"),
    ("exam registration fee", "Education"),
    ("coding bootcamp payment", "Education"),

    # Entertainment
    ("netflix subscription", "Entertainment"),
    ("spotify premium", "Entertainment"),
    ("cgv cinema ticket", "Entertainment"),
    ("lotte cinema movie", "Entertainment"),
    ("steam game purchase", "Entertainment"),
    ("karaoke evening", "Entertainment"),
    ("concert ticket", "Entertainment"),
    ("youtube premium subscription", "Entertainment"),
    ("gaming top up", "Entertainment"),
    ("event ticket purchase", "Entertainment"),

    # Savings
    ("bank savings deposit", "Savings"),
    ("tiet kiem gui ngan hang", "Savings"),
    ("investment fund transfer", "Savings"),
    ("stock purchase broker", "Savings"),
    ("chung khoan dau tu", "Savings"),

    # General (catch-all)
    ("atm withdrawal cash", "General"),
    ("bank transfer payment", "General"),
    ("miscellaneous payment", "General"),
    ("unknown transaction", "General"),
    ("other expense", "General"),
]


def train():
    texts  = [d for d, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    print(f"[Categorizer] Training on {len(texts)} examples across {len(set(labels))} categories...")
    clf = TransactionClassifier()
    clf.train(texts, labels)
    clf.save()
    print("[Categorizer] Saved -> categorizer/saved/classifier.pkl")


if __name__ == "__main__":
    train()
