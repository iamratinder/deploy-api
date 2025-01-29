from pydantic import BaseModel
class info_data(BaseModel):
    age:int
    income: int
    ownership: str
    employment_len: int
    loan_intent: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cred_hist_len: int