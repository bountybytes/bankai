from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AccountDetails(BaseModel):
    bank_name:        str = ""
    branch_name:      str = ""
    bank_type:        str = ""
    account_number:   str = ""
    account_holder:   str = ""
    address:          str = ""
    pan:              str = ""
    mobile:           str = ""
    email:            str = ""
    ifsc_code:        str = ""
    micr_code:        str = ""
    account_type:     str = ""
    statement_from:   str = ""
    statement_to:     str = ""
    opening_balance:  str = ""
    closing_balance:  str = ""
    currency:         str = "INR"


class Transaction(BaseModel):
    date:             str = ""
    value_date:       str = ""
    description:      str = ""
    narration:        str = ""
    cheque_no:        str = ""
    reference_no:     str = ""
    transaction_type: str = ""   # "DEBIT" | "CREDIT"
    debit:            str = ""
    credit:           str = ""
    balance:          str = ""
    branch_code:      str = ""
    remarks:          str = ""
    category:         str = ""   # filled by categorizer.py


class Summary(BaseModel):
    total_transactions: int   = 0
    total_debit:        float = 0.0
    total_credit:       float = 0.0
    net_flow:           float = 0.0


class ParseResponse(BaseModel):
    account_details:  AccountDetails
    transactions:     List[Transaction]
    summary:          Summary
    source_file:      str
    used_image_mode:  bool = False
    timings:          Dict[str, Any] = Field(default_factory=dict)