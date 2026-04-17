import re
from typing import List, Tuple

CATEGORY_RULES: List[Tuple[str, str]] = [
    (r"salary|sal\b|payroll|stipend|wages|sal\.cr|ctc", "Income"),
    (r"interest|int\.pd|int\.cr|savings?\s?interest|int earned", "Interest Earned"),
    (r"emi\b|loan|home\s?loan|car\s?loan|personal\s?loan|ach[\s\-]dr|nach|equated", "Loan/EMI"),
    (r"insurance|lic\b|policy|premium|hdfc\s?life|sbi\s?life|bajaj|tata\s?aia|new\s?india|star\s?health", "Insurance"),
    (r"mutual\s?fund|sip\b|zerodha|groww|upstox|invest|stock|demat|nse|bse|ppf|nps\b|elss", "Investment"),
    (r"credit\s?card|card\s?payment|cc\s?bill|card\s?charges|hdfc\s?cc|icici\s?cc|axis\s?cc", "Credit Card Payment"),
    (r"tax|gst|income\s?tax|tds\b|advance\s?tax|it\s?dept|incometax", "Tax"),
    (r"upi|imps|neft|rtgs|transfer|trf|p2p|sent\s?to|received\s?from|phonepay|gpay|paytm|bhim", "Transfer"),
    (r"atm|cash\s?with(draw)?|withdrawal|cdm\b", "Cash Withdrawal"),
    (r"swiggy|zomato|uber\s?eat|dunzo|food|restaurant|cafe|kfc|mcdonalds|domino|blinkit|zesty|eatclub", "Food & Dining"),
    (r"big\s?bazaar|dmart|reliance\s?smart|grocer|supermark|grocery|zepto|instamart|jiomart|more\s?retail", "Groceries"),
    (r"electricity|water\s?bill|gas\s?bill|jio|airtel|vi\b|vodafone|bsnl|broadband|internet|apepdcl|bescom|msedcl|tpddl", "Utilities"),
    (r"rent|pg\b|hostel|lease|landlord|maintenance|society|hoa\b", "Rent/Housing"),
    (r"ola\b|uber\b|rapido|auto|taxi|metro|irctc|railway|petrol|fuel|fastag|toll|parking|cab\b", "Transport"),
    (r"amazon|flipkart|myntra|meesho|ajio|nykaa|snapdeal|tata\s?cliq|shopsy|reliance\s?digital", "Shopping"),
    (r"netflix|prime|hotstar|spotify|movie|cinema|pvr|inox|steam|disney|zee5|sonyliv", "Entertainment"),
    (r"hospital|clinic|pharmacy|apollo|fortis|health|doctor|diagnostic|pharma|medplus|multispec|1mg|netmeds", "Healthcare"),
    (r"school|college|tuition|course|udemy|education|fees|cbse|univ|byju|vedantu|unacademy", "Education"),
    (r"hotel|resort|airbnb|oyo|goibibo|makemytrip|travel|flight|indigo|spice|vistara|cleartrip", "Travel"),
    (r"recharge|dth\b|tatasky|topup|d2h|sun\s?direct|dish\s?tv", "Recharge"),
    (r"charges|fee\b|penalty|service\s?charge|ecs\s?txn|bank\s?charge|gst\s?chg|annual\s?fee|sms\s?chg", "Bank Charges"),
    (r"dep\s?tfr|deposit|credited|cr\b", "Deposit"),
    (r"wdl\s?tfr|withdrew|dr\b", "Withdrawal"),
]


def categorize(description: str) -> str:
    d = description.lower()
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, d):
            return category
    return "Other"


def categorize_transactions(transactions: list) -> list:
    for txn in transactions:
        text = f"{txn.get('description', '')} {txn.get('narration', '')} {txn.get('remarks', '')}"
        txn["category"] = categorize(text)
    return transactions
