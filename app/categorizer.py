import re
from typing import List, Tuple

CATEGORY_RULES: List[Tuple] = [
    (r"salary|sal\b|payroll|stipend|wages",                                               "Income"),
    (r"upi|imps|neft|rtgs|transfer|trf",                                                  "Transfer"),
    (r"atm|cash\s?withdraw|withdrawal",                                                   "Cash Withdrawal"),
    (r"swiggy|zomato|uber\s?eat|dunzo|food|restaurant|cafe|kfc|mcdonalds|domino|blinkit", "Food & Dining"),
    (r"big\s?bazaar|dmart|reliance\s?smart|grocer|supermark|grocery",                    "Groceries"),
    (r"electricity|water\s?bill|gas\s?bill|jio|airtel|vodafone|bsnl|broadband|apepdcl",  "Utilities"),
    (r"rent|pg\b|hostel|lease|landlord|maintenance",                                      "Rent/Housing"),
    (r"ola\b|uber\b|rapido|auto|taxi|metro|irctc|railway|petrol|fuel|fastag",             "Transport"),
    (r"amazon|flipkart|myntra|meesho|ajio|shopping|nykaa|snapdeal",                       "Shopping"),
    (r"netflix|prime|hotstar|spotify|movie|cinema|pvr|inox|steam",                        "Entertainment"),
    (r"hospital|clinic|pharmacy|apollo|fortis|health|doctor|diagnostic|pharma|multispec", "Healthcare"),
    (r"school|college|tuition|course|udemy|education|fees|cbse",                          "Education"),
    (r"insurance|lic\b|policy|premium|hdfc\s?life|sbi\s?life|bajaj",                     "Insurance"),
    (r"emi\b|loan|home\s?loan|car\s?loan|personal\s?loan|ach-dr",                        "Loan/EMI"),
    (r"mutual\s?fund|sip\b|zerodha|groww|upstox|invest|stock|demat",                     "Investment"),
    (r"credit\s?card|card\s?payment|cc\s?bill|card charges",                              "Bank Charges"),
    (r"interest|int\.pd|int\.cr|savings\s?interest",                                      "Interest Earned"),
    (r"charges|fee\b|penalty|service\s?charge|ecs\s?txn",                                 "Bank Charges"),
    (r"hotel|resort|airbnb|oyo|goibibo|makemytrip|travel",                                "Travel"),
    (r"recharge|dth\b|tatasky|topup",                                                     "Recharge"),
]

def categorize(description: str) -> str:
    d = description.lower()
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, d):
            return category
    return "Other"

def categorize_transactions(transactions: list) -> list:
    for txn in transactions:
        txn["category"] = categorize(txn.get("description", ""))
    return transactions