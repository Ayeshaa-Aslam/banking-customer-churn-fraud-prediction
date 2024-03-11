"""
02_banking_data_quality.py
Reusable data quality checks for banking_insights.db
Run: py scripts/02_banking_data_quality.py
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = "database/banking_insights.db"


def load_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bank = pd.read_sql("SELECT * FROM banking_customers", conn)
    fraud = pd.read_sql("SELECT * FROM fraud_data", conn)
    links = pd.read_sql("SELECT * FROM customer_links", conn)
    return bank, fraud, links


def missing_summary(df: pd.DataFrame) -> pd.Series:
    s = df.isna().sum()
    return s[s > 0].sort_values(ascending=False)


def domain_checks_bank(bank: pd.DataFrame) -> dict[str, pd.DataFrame]:
    issues: dict[str, pd.DataFrame] = {}
    issues['credit_score_out_of_range'] = bank[(bank['credit_score'] < 300) | (bank['credit_score'] > 900)]
    issues['age_out_of_range'] = bank[(bank['age'] < 16) | (bank['age'] > 100)]
    issues['tenure_out_of_range'] = bank[(bank['tenure'] < 0) | (bank['tenure'] > 50)]
    issues['balance_negative'] = bank[bank['balance'] < 0]
    issues['products_number_out_of_range'] = bank[(bank['products_number'] < 0) | (bank['products_number'] > 10)]
    issues['credit_card_not_binary'] = bank[~bank['credit_card'].isin([0, 1])]
    issues['active_member_not_binary'] = bank[~bank['active_member'].isin([0, 1])]
    issues['estimated_salary_negative'] = bank[bank['estimated_salary'] < 0]
    issues['churn_not_binary'] = bank[~bank['churn'].isin([0, 1])]
    issues['country_invalid'] = bank[~bank['country'].isin(['France', 'Germany', 'Spain'])]
    issues['gender_invalid'] = bank[~bank['gender'].isin(['Male', 'Female'])]
    return issues


def iqr_outlier_counts(bank: pd.DataFrame) -> pd.Series:
    def iqr_count(series: pd.Series) -> int:
        arr = series.dropna().values
        if arr.size == 0:
            return 0
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((series < lower) | (series > upper)).sum())

    numeric_cols = [
        'credit_score', 'age', 'tenure', 'balance', 'products_number',
        'estimated_salary', 'balance_to_salary_ratio', 'products_per_year'
    ]
    present = [c for c in numeric_cols if c in bank.columns]
    return pd.Series({c: iqr_count(bank[c]) for c in present}, name='iqr_outliers')


def fraud_checks(fraud: pd.DataFrame) -> dict[str, pd.DataFrame]:
    issues: dict[str, pd.DataFrame] = {}
    issues['amount_negative'] = fraud[fraud['Amount'] < 0]
    issues['time_negative'] = fraud[fraud['Time'] < 0]
    issues['class_not_binary'] = fraud[~fraud['Class'].isin([0, 1])]
    return issues


def main() -> None:
    if not Path(DB_PATH).exists():
        raise SystemExit(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    bank, fraud, links = load_tables(conn)

    print("== TABLE SHAPES ==")
    print(f"banking_customers: {bank.shape}")
    print(f"fraud_data: {fraud.shape}")
    print(f"customer_links: {links.shape}")

    print("\n== MISSING VALUES (non-zero only) ==")
    mb, mf = missing_summary(bank), missing_summary(fraud)
    print("banking_customers:\n", mb if not mb.empty else "None")
    print("fraud_data:\n", mf if not mf.empty else "None")

    print("\n== DUPLICATES & KEYS ==")
    print("bank duplicates:", int(bank.duplicated().sum()))
    print("fraud duplicates:", int(fraud.duplicated().sum())))
    print("unique banking customer_id:", bank['customer_id'].nunique())
    print("unique fraud customer_id:", fraud['customer_id'].nunique())

    print("\n== BANK DOMAIN ISSUES (counts) ==")
    bank_issues = domain_checks_bank(bank)
    print({k: len(v) for k, v in bank_issues.items()})

    print("\n== BANK NUMERIC IQR OUTLIERS (counts) ==")
    print(iqr_outlier_counts(bank))

    print("\n== FRAUD CHECKS (counts) ==")
    fissues = fraud_checks(fraud)
    print({k: len(v) for k, v in fissues.items()})

    print("\n== FRAUD SUMMARY ==")
    fraud_rate = fraud['Class'].mean() * 100
    print(f"Fraud rate: {fraud_rate:.2f}%")

    print("\n== LINK COVERAGE ==")
    bank_ids, fraud_ids = set(bank['customer_id']), set(fraud['customer_id'])
    link_bank_cov = (links['banking_customer_id'].isin(bank_ids)).mean() * 100
    link_fraud_cov = (links['fraud_customer_id'].isin(fraud_ids)).mean() * 100
    print(f"Links: {len(links):,}")
    print(f"Banking coverage: {link_bank_cov:.1f}%  |  Fraud coverage: {link_fraud_cov:.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
