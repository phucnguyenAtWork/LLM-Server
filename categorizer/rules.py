"""
Rule-based category patterns.
Keys must match category names in the DB `categories` table.
Order within each list doesn't matter — all are checked via regex OR.
"""

import re

CATEGORY_RULES: dict[str, list[str]] = {
    "Food": [
        r"grab\s*food", r"grabfood", r"shopee\s*food", r"baemin",
        r"gofood", r"now\.vn", r"kfc", r"mcdonald", r"lotteria",
        r"jollibee", r"pizza", r"domino", r"highlands", r"starbucks",
        r"phuc\s*long", r"coffee\s*house", r"trung\s*nguyen",
        r"restaurant", r"cafe", r"quan\s*an", r"com\s*tam",
        r"bun\s*bo", r"pho", r"banh", r"food", r"eat",
    ],
    "Transport": [
        r"grab\s*car", r"grab\s*bike", r"grab\b", r"be\s*app", r"\bbe\b",
        r"xanh\s*sm", r"taxi", r"parking", r"bai\s*xe",
        r"xang", r"petrol", r"fuel", r"bus", r"xe\s*buyt",
        r"ve\s*xe", r"tram", r"metro",
    ],
    "Shopping": [
        r"shopee\b", r"lazada", r"tiki\b", r"amazon", r"vinmart",
        r"winmart", r"co\.?op\s*mart", r"coopmart", r"circle\s*k",
        r"7\s*eleven", r"7-eleven", r"guardian", r"watsons",
        r"fahasa", r"mall", r"siêu\s*thị", r"sieu\s*thi",
    ],
    "Bills": [
        r"fpt\b", r"viettel\b", r"vnpt\b", r"evn\b",
        r"electric", r"tien\s*dien", r"hoa\s*don\s*dien",
        r"water\s*bill", r"tien\s*nuoc",
        r"internet\s*bill", r"cuoc\s*mang",
        r"thanh\s*toan\s*hoa\s*don", r"payment\s*bill",
        r"rent", r"tien\s*nha",
    ],
    "Health": [
        r"pharmacity", r"long\s*chau", r"pharmacy", r"nha\s*thuoc",
        r"hospital", r"benh\s*vien", r"clinic", r"phong\s*kham",
        r"thuoc\b", r"medical", r"y\s*te", r"suc\s*khoe",
    ],
    "Education": [
        r"tuition", r"hoc\s*phi", r"truong\b", r"school", r"university",
        r"khoa\s*hoc", r"course", r"udemy", r"coursera",
        r"sach\b", r"fahasa", r"book\b",
    ],
    "Entertainment": [
        r"netflix", r"spotify", r"cgv\b", r"lotte\s*cinema",
        r"steam\b", r"youtube\s*premium", r"game\b", r"karaoke",
        r"cinema", r"rap\s*chieu\s*phim", r"concert", r"su\s*kien",
    ],
    "Savings": [
        r"tiet\s*kiem", r"saving", r"deposit", r"dau\s*tu",
        r"investment", r"chung\s*khoan", r"stock",
    ],
}

# Compiled for speed
_COMPILED: dict[str, re.Pattern] = {
    cat: re.compile("|".join(patterns), re.IGNORECASE)
    for cat, patterns in CATEGORY_RULES.items()
}


def rule_predict(text: str) -> tuple[str | None, float]:
    """
    Returns (category, confidence) using regex rules.
    confidence=1.0 means a hard rule match; None means no match.
    """
    for category, pattern in _COMPILED.items():
        if pattern.search(text):
            return category, 1.0
    return None, 0.0
