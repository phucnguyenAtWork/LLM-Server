"""
NLP Pipeline — Transaction Description Normalizer
==================================================
Cleans raw bank/e-wallet transaction strings into structured, readable output.

Examples:
    "GRAB*FOOD-HCM-9234817"  → merchant: "Grab Food",  cleaned: "grab food hcm"
    "MOMO CK DEN 0912345678" → merchant: "MoMo",        cleaned: "momo ck den"
    "ATM WD 123456 VCB"      → merchant: "ATM Withdrawal", cleaned: "atm wd vcb"
"""

import re

# ── Cleaning patterns applied in order ───────────────────────────────────────
_CLEAN_STEPS = [
    (r'\b\d{5,}\b',   ''),    # strip long numeric IDs (transaction codes, phone numbers)
    (r'[*_/\\|]+',    ' '),   # replace special separators with space
    (r'-+',           ' '),   # hyphens → space
    (r'\s{2,}',       ' '),   # collapse multiple spaces
]

# ── Known merchant normalizations (lowercase key → display name) ─────────────
MERCHANT_MAP = {
    # Food & Drink
    'grab food':       'Grab Food',
    'grabfood':        'Grab Food',
    'shopee food':     'Shopee Food',
    'baemin':          'Baemin',
    'gofood':          'GoFood',
    'now.vn':          'Now.vn',
    'kfc':             'KFC',
    'mcdonalds':       "McDonald's",
    'mcdonald':        "McDonald's",
    'lotteria':        'Lotteria',
    'jollibee':        'Jollibee',
    'pizza hut':       'Pizza Hut',
    'dominos':         "Domino's",
    'highlands':       'Highlands Coffee',
    'starbucks':       'Starbucks',
    'phuc long':       'Phuc Long',
    'the coffee house':'The Coffee House',
    'trung nguyen':    'Trung Nguyên',
    # Transport
    'grab car':        'Grab Car',
    'grab bike':       'Grab Bike',
    'grab':            'Grab',
    'be app':          'Be App',
    'be ':             'Be App',
    'xanh sm':         'Xanh SM',
    # Shopping
    'shopee':          'Shopee',
    'lazada':          'Lazada',
    'tiki':            'Tiki',
    'amazon':          'Amazon',
    'vinmart':         'VinMart',
    'winmart':         'WinMart',
    'co.opmart':       "Co.opMart",
    'coopmart':        "Co.opMart",
    'circle k':        'Circle K',
    '7 eleven':        '7-Eleven',
    '7-eleven':        '7-Eleven',
    'guardian':        'Guardian',
    'watsons':         'Watsons',
    'fahasa':          'Fahasa',
    # Bills & Utilities
    'fpt':             'FPT Internet',
    'viettel':         'Viettel',
    'vnpt':            'VNPT',
    'evn':             'EVN (Electricity)',
    # E-wallets & Payment
    'momo':            'MoMo',
    'vnpay':           'VNPay',
    'zalopay':         'ZaloPay',
    'payoo':           'Payoo',
    # Entertainment
    'netflix':         'Netflix',
    'spotify':         'Spotify',
    'cgv':             'CGV Cinema',
    'lotte cinema':    'Lotte Cinema',
    'steam':           'Steam',
    'youtube':         'YouTube Premium',
    # Health
    'pharmacity':      'Pharmacity',
    'long chau':       'Long Châu Pharmacy',
    # Banking / ATM
    'atm wd':          'ATM Withdrawal',
    'atm withdrawal':  'ATM Withdrawal',
}

# Sort by key length descending so longer matches take priority
_MERCHANT_MAP_SORTED = sorted(MERCHANT_MAP.items(), key=lambda x: len(x[0]), reverse=True)


def clean_description(text: str) -> str:
    """Lowercase, strip IDs, normalize separators."""
    if not text:
        return ''
    result = text.lower().strip()
    for pattern, replacement in _CLEAN_STEPS:
        result = re.sub(pattern, replacement, result)
    return result.strip()


def extract_merchant(text: str) -> str:
    """
    Match cleaned text against known merchant patterns.
    Falls back to capitalizing the first 3 meaningful words.
    """
    cleaned = clean_description(text)
    for key, display in _MERCHANT_MAP_SORTED:
        if key in cleaned:
            return display
    # Fallback: capitalize first few words, skip noise words
    noise = {'the', 'and', 'at', 'in', 'on', 'of', 'ck', 'den', 'tu', 'cho'}
    words = [w for w in cleaned.split() if w not in noise and len(w) > 1]
    return ' '.join(w.capitalize() for w in words[:3]) if words else text.strip()


def tokenize(text: str) -> list[str]:
    """Return meaningful tokens (length > 2, no pure digits)."""
    cleaned = clean_description(text)
    return [t for t in cleaned.split() if len(t) > 2 and not t.isdigit()]


def normalize_transaction(description: str) -> dict:
    """
    Full NLP pipeline for a raw transaction description.

    Returns:
        {
            original  : str  — untouched input
            cleaned   : str  — lowercased, stripped of IDs/separators
            merchant  : str  — human-readable merchant name
            tokens    : list — meaningful tokens for ML features
        }
    """
    return {
        'original': description,
        'cleaned':  clean_description(description),
        'merchant': extract_merchant(description),
        'tokens':   tokenize(description),
    }
