import re
from typing import List, Tuple

# ── Category rules ─────────────────────────────────────────────────────────────
# Order matters — first match wins.
CATEGORY_RULES: List[Tuple[str, str]] = [
    (r"salary|sal\b|payroll|stipend|wages",                                                           "Income"),
    (r"interest|int\.pd|int\.cr|savings?\s?interest",                                                 "Interest Earned"),
    (r"emi\b|loan|home\s?loan|car\s?loan|personal\s?loan|ach[\s\-]dr|nach",                          "Loan/EMI"),
    (r"insurance|lic\b|policy|premium|hdfc\s?life|sbi\s?life|bajaj|tata\s?aia",                      "Insurance"),
    (r"mutual\s?fund|sip\b|zerodha|groww|upstox|invest|stock|demat|nse|bse",                         "Investment"),
    (r"credit\s?card|card\s?payment|cc\s?bill|card\s?charges",                                       "Credit Card Payment"),
    (r"tax|gst|income\s?tax|tds\b|advance\s?tax",                                                    "Tax"),
    (r"upi|imps|neft|rtgs|transfer|trf",                                                              "Transfer"),
    (r"atm|cash\s?withdraw|withdrawal",                                                               "Cash Withdrawal"),
    (r"swiggy|zomato|uber\s?eat|dunzo|food|restaurant|cafe|kfc|mcdonalds|domino|blinkit|zesty",      "Food & Dining"),
    (r"big\s?bazaar|dmart|reliance\s?smart|grocer|supermark|grocery|blinkit|zepto|instamart",        "Groceries"),
    (r"electricity|water\s?bill|gas\s?bill|jio|airtel|vi\b|vodafone|bsnl|broadband|internet|apepdcl","Utilities"),
    (r"rent|pg\b|hostel|lease|landlord|maintenance",                                                  "Rent/Housing"),
    (r"ola\b|uber\b|rapido|auto|taxi|metro|irctc|railway|petrol|fuel|fastag|toll",                   "Transport"),
    (r"amazon|flipkart|myntra|meesho|ajio|shopping|nykaa|snapdeal|tata\s?cliq",                      "Shopping"),
    (r"netflix|prime|hotstar|spotify|movie|cinema|pvr|inox|steam|disney",                            "Entertainment"),
    (r"hospital|clinic|pharmacy|apollo|fortis|health|doctor|diagnostic|pharma|medplus|multispec",    "Healthcare"),
    (r"school|college|tuition|course|udemy|education|fees|cbse|univ",                                "Education"),
    (r"hotel|resort|airbnb|oyo|goibibo|makemytrip|travel|flight|indigo|spice",                       "Travel"),
    (r"recharge|dth\b|tatasky|topup|d2h",                                                            "Recharge"),
    (r"charges|fee\b|penalty|service\s?charge|ecs\s?txn|bank\s?charge|gst\s?chg",                   "Bank Charges"),
]


def categorize(description: str) -> str:
    d = description.lower()
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, d):
            return category
    return "Other"


def categorize_transactions(transactions: list) -> list:
    for txn in transactions:
        # Combine description + narration for better matching
        text = f"{txn.get('description', '')} {txn.get('narration', '')}"
        txn["category"] = categorize(text)
    return transactions