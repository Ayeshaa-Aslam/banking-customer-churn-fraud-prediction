"""
02_banking_data_exploration.py
Banking EDA: reproduce key queries and export PNGs/CSVs

Run:
  py scripts/02_banking_data_exploration.py

Outputs (written to analysis/):
  - churn_by_credit_score_tier.csv /.png
  - churn_by_country.csv /.png
  - churn_by_customer_value_tier.csv /.png
  - churn_by_products_number.csv /.png
  - churn_by_active_member.csv /.png
  - heatmap_country_by_credit_score_tier.png
  - heatmap_country_by_customer_value_tier.png
  - crosstab CSVs for country×credit_score_tier, country×customer_value_tier, credit_score_tier×customer_value_tier
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DB_PATH = "database/banking_insights.db"
OUT_DIR = Path("analysis")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_table(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)


def save_barplot(df: pd.DataFrame, x: str, y: str, title: str, fname: str, order=None) -> None:
    plt.figure(figsize=(8, 5))
    if order is None:
        ax = sns.barplot(data=df, x=x, y=y)
    else:
        ax = sns.barplot(data=df, x=x, y=y, order=order)
    ax.set_title(title)
    ax.set_ylabel("Churn rate")
    ax.set_xlabel(x)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150)
    plt.close()


def save_heatmap(pivot_df: pd.DataFrame, title: str, fname: str) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn_r")
    ax.set_title(title)
    ax.set_xlabel(pivot_df.columns.name or "")
    ax.set_ylabel(pivot_df.index.name or "")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_out_dir()

    # 1) Churn by credit score tier
    credit_sql = """
    SELECT credit_score_tier, COUNT(*) AS cnt, AVG(churn) AS churn_rate
    FROM banking_customers
    GROUP BY credit_score_tier
    ORDER BY CASE credit_score_tier
      WHEN 'Poor' THEN 1 WHEN 'Fair' THEN 2 WHEN 'Good' THEN 3 WHEN 'Excellent' THEN 4 END
    """
    churn_by_credit = read_table(credit_sql)
    save_table(churn_by_credit, "churn_by_credit_score_tier")
    save_barplot(
        churn_by_credit,
        x="credit_score_tier",
        y="churn_rate",
        title="Churn rate by credit score tier",
        fname="churn_by_credit_score_tier",
        order=["Poor", "Fair", "Good", "Excellent"],
    )

    # 2) Churn by country
    country_sql = """
    SELECT country, COUNT(*) AS cnt, AVG(churn) AS churn_rate
    FROM banking_customers
    GROUP BY country
    ORDER BY churn_rate DESC
    """
    churn_by_country = read_table(country_sql)
    save_table(churn_by_country, "churn_by_country")
    save_barplot(
        churn_by_country,
        x="country",
        y="churn_rate",
        title="Churn rate by country",
        fname="churn_by_country",
    )

    # 3) Churn by customer value tier
    value_sql = """
    SELECT customer_value_tier, COUNT(*) AS cnt, AVG(churn) AS churn_rate
    FROM banking_customers
    GROUP BY customer_value_tier
    ORDER BY CASE customer_value_tier
      WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 WHEN 'Premium' THEN 4 END
    """
    churn_by_value = read_table(value_sql)
    save_table(churn_by_value, "churn_by_customer_value_tier")
    save_barplot(
        churn_by_value,
        x="customer_value_tier",
        y="churn_rate",
        title="Churn rate by customer value tier",
        fname="churn_by_customer_value_tier",
        order=["Low", "Medium", "High", "Premium"],
    )

    # 4) Products and engagement impacts
    products_sql = """
    SELECT products_number, COUNT(*) AS cnt, AVG(churn) AS churn_rate
    FROM banking_customers
    GROUP BY products_number
    ORDER BY products_number
    """
    churn_by_products = read_table(products_sql)
    save_table(churn_by_products, "churn_by_products_number")
    save_barplot(
        churn_by_products,
        x="products_number",
        y="churn_rate",
        title="Churn rate by number of products",
        fname="churn_by_products_number",
    )

    active_sql = """
    SELECT active_member, COUNT(*) AS cnt, AVG(churn) AS churn_rate
    FROM banking_customers
    GROUP BY active_member
    ORDER BY active_member
    """
    churn_by_active = read_table(active_sql)
    save_table(churn_by_active, "churn_by_active_member")
    # Map 0/1 to labels for nicer chart
    tmp = churn_by_active.copy()
    tmp["active_member"] = tmp["active_member"].map({0: "Inactive", 1: "Active"})
    save_barplot(
        tmp,
        x="active_member",
        y="churn_rate",
        title="Churn rate by engagement (active vs inactive)",
        fname="churn_by_active_member",
        order=["Inactive", "Active"],
    )

    # 5) Cross-tabs for inter-relationships
    # country × credit_score_tier
    ctab_country_credit = read_table(
        """
        SELECT country, credit_score_tier, AVG(churn) AS churn_rate
        FROM banking_customers
        GROUP BY country, credit_score_tier
        """
    )
    save_table(ctab_country_credit, "crosstab_country_by_credit_score_tier")
    pivot_cc = ctab_country_credit.pivot(index="country", columns="credit_score_tier", values="churn_rate")
    # Ensure all tiers exist as columns and order them; fill gaps if any
    pivot_cc = pivot_cc.reindex(columns=["Poor", "Fair", "Good", "Excellent"]).fillna(0)
    save_heatmap(pivot_cc, "Churn rate: country × credit score tier", "heatmap_country_by_credit_score_tier")

    # country × customer_value_tier
    ctab_country_value = read_table(
        """
        SELECT country, customer_value_tier, AVG(churn) AS churn_rate
        FROM banking_customers
        GROUP BY country, customer_value_tier
        """
    )
    save_table(ctab_country_value, "crosstab_country_by_customer_value_tier")
    pivot_cv = ctab_country_value.pivot(index="country", columns="customer_value_tier", values="churn_rate").loc[:, ["Low", "Medium", "High", "Premium"]]
    save_heatmap(pivot_cv, "Churn rate: country × customer value tier", "heatmap_country_by_customer_value_tier")

    # credit_score_tier × customer_value_tier (how wealth vs credit interacts)
    ctab_credit_value = read_table(
        """
        SELECT credit_score_tier, customer_value_tier, AVG(churn) AS churn_rate
        FROM banking_customers
        GROUP BY credit_score_tier, customer_value_tier
        """
    )
    save_table(ctab_credit_value, "crosstab_credit_score_tier_by_customer_value_tier")
    pivot_cv2 = ctab_credit_value.pivot(index="credit_score_tier", columns="customer_value_tier", values="churn_rate").loc[["Poor", "Fair", "Good", "Excellent"], ["Low", "Medium", "High", "Premium"]]
    save_heatmap(pivot_cv2, "Churn rate: credit score tier × customer value tier", "heatmap_credit_score_tier_by_customer_value_tier")

    print("Saved analysis to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()


