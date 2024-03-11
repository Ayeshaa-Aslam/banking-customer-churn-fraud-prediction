"""
02_banking_financial_behavior_analysis.py
Complete financial behavior analysis for tasks 2.5-2.7

Analyzes:
- 2.5: Account balance distribution, high-value customers, zero-balance patterns
- 2.6: Active member patterns, credit card ownership, tenure analysis  
- 2.7: Salary segmentation, wealth tiers, income vs balance correlation

Run: py scripts/02_banking_financial_behavior_analysis.py

Outputs (analysis/):
- balance_analysis.csv/.png
- high_value_customers.csv/.png  
- zero_balance_analysis.csv/.png
- active_vs_inactive.csv/.png
- credit_card_ownership.csv/.png
- tenure_analysis.csv/.png
- salary_segmentation.csv/.png
- wealth_correlation_matrix.csv/.png
- balance_vs_salary_scatter.png
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "database/banking_insights.db"
OUT_DIR = Path("analysis")

def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_bank():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM banking_customers", conn)

def save_table(df, name):
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)

def save_plot(title, fname, figsize=(8,5)):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    ensure_out_dir()
    bank = load_bank()
    
    print("🏦 FINANCIAL BEHAVIOR ANALYSIS")
    print("="*50)
    
    # ============= 2.5: ACCOUNT BALANCE ANALYSIS =============
    print("\n📊 2.5: Account Balance Analysis")
    
    # Balance distribution by churn
    balance_bins = [0, 0.1, 25000, 50000, 100000, 150000, float('inf')]
    balance_labels = ['Zero', 'Low (0-25K)', 'Medium (25-50K)', 'High (50-100K)', 'Premium (100-150K)', 'Ultra (150K+)']
    bank['balance_tier'] = pd.cut(bank['balance'], bins=balance_bins, labels=balance_labels, right=False)
    
    balance_analysis = (bank.groupby('balance_tier', observed=False)['churn']
                       .agg(['mean', 'count']).reset_index()
                       .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    
    save_table(balance_analysis, 'balance_analysis')
    
    plt.figure(figsize=(10,5))
    sns.barplot(data=balance_analysis, x='balance_tier', y='churn_rate')
    plt.xticks(rotation=45)
    save_plot('Churn Rate by Balance Tier', 'balance_analysis')
    
    # High-value customer analysis (>100K balance)
    high_value_threshold = 100000
    bank['is_high_balance'] = (bank['balance'] > high_value_threshold).astype(int)
    
    high_value_analysis = (bank.groupby('is_high_balance')['churn']
                          .agg(['mean', 'count']).reset_index()
                          .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    high_value_analysis['segment'] = high_value_analysis['is_high_balance'].map({0: 'Regular (<100K)', 1: 'High-Value (>100K)'})
    
    save_table(high_value_analysis, 'high_value_customers')
    
    plt.figure(figsize=(6,4))
    sns.barplot(data=high_value_analysis, x='segment', y='churn_rate')
    save_plot('Churn Rate: High-Value vs Regular Customers', 'high_value_customers')
    
    # Zero-balance customer patterns
    zero_balance = bank[bank['balance'] == 0].copy()
    zero_analysis = {
        'total_zero_balance': len(zero_balance),
        'zero_balance_churn_rate': zero_balance['churn'].mean(),
        'zero_balance_pct_of_total': len(zero_balance) / len(bank) * 100,
        'avg_products_zero_balance': zero_balance['products_number'].mean(),
        'avg_tenure_zero_balance': zero_balance['tenure'].mean()
    }
    
    zero_df = pd.DataFrame([zero_analysis])
    save_table(zero_df, 'zero_balance_analysis')
    
    print(f"   • Zero-balance customers: {zero_analysis['total_zero_balance']:,} ({zero_analysis['zero_balance_pct_of_total']:.1f}%)")
    print(f"   • Zero-balance churn rate: {zero_analysis['zero_balance_churn_rate']:.1%}")
    
    # ============= 2.6: CUSTOMER ENGAGEMENT METRICS =============
    print("\n👤 2.6: Customer Engagement Analysis")
    
    # Active vs inactive patterns (already have active_member)
    active_analysis = (bank.groupby('active_member')['churn']
                      .agg(['mean', 'count']).reset_index()
                      .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    active_analysis['segment'] = active_analysis['active_member'].map({0: 'Inactive', 1: 'Active'})
    
    save_table(active_analysis, 'active_vs_inactive')
    
    plt.figure(figsize=(6,4))
    sns.barplot(data=active_analysis, x='segment', y='churn_rate')
    save_plot('Churn Rate: Active vs Inactive Members', 'active_vs_inactive')
    
    # Credit card ownership impact
    card_analysis = (bank.groupby('credit_card')['churn']
                    .agg(['mean', 'count']).reset_index()
                    .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    card_analysis['segment'] = card_analysis['credit_card'].map({0: 'No Credit Card', 1: 'Has Credit Card'})
    
    save_table(card_analysis, 'credit_card_ownership')
    
    plt.figure(figsize=(6,4))
    sns.barplot(data=card_analysis, x='segment', y='churn_rate')
    save_plot('Churn Rate by Credit Card Ownership', 'credit_card_ownership')
    
    # Tenure analysis (years with bank)
    tenure_bins = [0, 2, 4, 6, 8, 12]
    tenure_labels = ['New (0-2yr)', 'Growing (2-4yr)', 'Established (4-6yr)', 'Mature (6-8yr)', 'Veteran (8yr+)']
    bank['tenure_tier'] = pd.cut(bank['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
    
    tenure_analysis = (bank.groupby('tenure_tier', observed=False)['churn']
                      .agg(['mean', 'count']).reset_index()
                      .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    
    save_table(tenure_analysis, 'tenure_analysis')
    
    plt.figure(figsize=(8,4))
    sns.barplot(data=tenure_analysis, x='tenure_tier', y='churn_rate')
    plt.xticks(rotation=45)
    save_plot('Churn Rate by Customer Tenure', 'tenure_analysis')
    
    # ============= 2.7: SALARY AND WEALTH SEGMENTATION =============
    print("\n💰 2.7: Salary and Wealth Analysis")
    
    # Salary impact on churn
    salary_bins = [0, 50000, 75000, 100000, 125000, float('inf')]
    salary_labels = ['Low (<50K)', 'Lower-Mid (50-75K)', 'Mid (75-100K)', 'Upper-Mid (100-125K)', 'High (125K+)']
    bank['salary_tier'] = pd.cut(bank['estimated_salary'], bins=salary_bins, labels=salary_labels, right=False)
    
    salary_analysis = (bank.groupby('salary_tier', observed=False)['churn']
                      .agg(['mean', 'count']).reset_index()
                      .rename(columns={'mean': 'churn_rate', 'count': 'customer_count'}))
    
    save_table(salary_analysis, 'salary_segmentation')
    
    plt.figure(figsize=(8,4))
    sns.barplot(data=salary_analysis, x='salary_tier', y='churn_rate')
    plt.xticks(rotation=45)
    save_plot('Churn Rate by Salary Tier', 'salary_segmentation')
    
    # Wealth correlation matrix
    wealth_cols = ['balance', 'estimated_salary', 'tenure', 'products_number', 'age', 'credit_score', 'churn']
    wealth_corr = bank[wealth_cols].corr()
    
    save_table(wealth_corr.reset_index(), 'wealth_correlation_matrix')
    
    plt.figure(figsize=(8,6))
    sns.heatmap(wealth_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    save_plot('Wealth & Behavior Correlation Matrix', 'wealth_correlation_matrix')
    
    # Balance vs salary scatter with churn overlay
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(bank['estimated_salary'], bank['balance'], 
                         c=bank['churn'], cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(scatter, label='Churn (0=No, 1=Yes)')
    plt.xlabel('Estimated Salary ($)')
    plt.ylabel('Account Balance ($)')
    save_plot('Balance vs Salary (Colored by Churn)', 'balance_vs_salary_scatter')
    
    # ============= SUMMARY INSIGHTS =============
    print("\n🎯 KEY INSIGHTS DISCOVERED:")
    print(f"   • High-value customers (>100K balance): {(bank['balance'] > 100000).sum():,} customers")
    print(f"   • Zero-balance churn rate: {zero_analysis['zero_balance_churn_rate']:.1%}")
    print(f"   • Active vs Inactive churn gap: {active_analysis.iloc[0]['churn_rate'] - active_analysis.iloc[1]['churn_rate']:.1%} points")
    print(f"   • Credit card owners: {(bank['credit_card'] == 1).sum():,} ({(bank['credit_card'] == 1).mean():.1%})")
    print(f"   • Balance-Salary correlation: {bank['balance'].corr(bank['estimated_salary']):.3f}")
    print(f"   • Tenure-Churn correlation: {bank['tenure'].corr(bank['churn']):.3f}")
    
    print(f"\n✅ All analysis saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
