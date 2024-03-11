"""
02_banking_data_exploration_demographics.py
Exports demographic and cross-sell EDA as CSVs and PNG charts.

Run:
  py scripts/02_banking_data_exploration_demographics.py

Outputs (analysis/):
  - age_churn.csv, age_churn.png
  - gender_churn.csv, gender_churn.png
  - heatmap_age_by_gender.csv, heatmap_age_by_gender.png
  - churn_by_products_binned.csv, churn_by_products_binned.png
  - heatmap_country_by_products_binned.csv, heatmap_country_by_products_binned.png
"""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DB_PATH = "database/banking_insights.db"
OUT_DIR = Path("analysis")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_bank() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM banking_customers", conn)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, fname: str, order=None, xlabel=None, ylabel="Churn rate") -> None:
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x=x, y=y, order=order)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150)
    plt.close()


def save_lineplot(df: pd.DataFrame, x: str, y: str, title: str, fname: str, xlabel=None, ylabel="Churn rate") -> None:
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x=x, y=y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150)
    plt.close()


def save_heatmap(pivot_df: pd.DataFrame, title: str, fname: str, xlabel: str = "", ylabel: str = "") -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn_r")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_out_dir()
    bank = load_bank()

    # Age bins
    age_bins = [18, 25, 30, 35, 40, 45, 50, 60, 100]
    age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-59", "60+"]
    bank["age_bin"] = pd.cut(bank["age"], bins=age_bins, labels=age_labels, right=False)

    # products_binned (1, 2, 3+)
    bank["products_binned"] = np.select(
        [bank["products_number"] == 1, bank["products_number"] == 2, bank["products_number"] >= 3],
        ["1", "2", "3_plus"],
        default="unknown",
    )

    # 1) Age churn
    age_churn = (
        bank.groupby("age_bin", observed=False)["churn"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "churn_rate", "count": "cnt"})
    )
    save_table(age_churn, "age_churn")
    save_lineplot(age_churn, x="age_bin", y="churn_rate", title="Churn rate by age bin", fname="age_churn", xlabel="Age bin")

    # 2) Gender churn
    gender_churn = (
        bank.groupby("gender", observed=False)["churn"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "churn_rate", "count": "cnt"})
    )
    save_table(gender_churn, "gender_churn")
    save_barplot(gender_churn, x="gender", y="churn_rate", title="Churn rate by gender", fname="gender_churn", xlabel="Gender")

    # 3) Age × gender heatmap
    age_gender = (
        bank.groupby(["age_bin", "gender"], observed=False)["churn"].mean().reset_index()
        .pivot(index="age_bin", columns="gender", values="churn").reindex(index=age_labels)
    )
    save_table(age_gender.reset_index().rename_axis(None, axis=1), "heatmap_age_by_gender")
    save_heatmap(age_gender, title="Churn rate: age bin × gender", fname="heatmap_age_by_gender", xlabel="Gender", ylabel="Age bin")

    # 4) Products binned churn
    products_binned = (
        bank.groupby("products_binned", observed=False)["churn"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "churn_rate", "count": "cnt"})
    )
    save_table(products_binned, "churn_by_products_binned")
    save_barplot(products_binned, x="products_binned", y="churn_rate", title="Churn rate by products (1, 2, 3+)", fname="churn_by_products_binned", order=["1", "2", "3_plus"], xlabel="Products")

    # 5) Country × products heatmap
    cxp = (
        bank.groupby(["country", "products_binned"], observed=False)["churn"].mean().reset_index()
        .pivot(index="country", columns="products_binned", values="churn").reindex(columns=["1", "2", "3_plus"]) 
    )
    save_table(cxp.reset_index().rename_axis(None, axis=1), "heatmap_country_by_products_binned")
    save_heatmap(cxp, title="Churn rate: country × products (1,2,3+)", fname="heatmap_country_by_products_binned", xlabel="Products", ylabel="Country")

    print("Saved analysis to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()


