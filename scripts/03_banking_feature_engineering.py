"""
03_banking_feature_engineering.py
Banking Feature Engineering - Transform raw data into model-ready features

Based on EDA insights:
- Engagement patterns are strongest predictors
- Balance tiers matter more than raw amounts  
- Geographic + demographic interactions are critical
- Zero-balance customers are a unique segment
- Cross-selling (1→2 products) has massive impact

Run: py scripts/03_banking_feature_engineering.py

Outputs:
- features/banking_features.csv (model-ready dataset)
- features/feature_importance_eda.csv (feature analysis)
- features/feature_correlation_matrix.csv (correlation analysis)
"""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "database/banking_insights.db"
OUT_DIR = Path("features")

def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_bank():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM banking_customers", conn)

def create_demographic_features(df):
    """Create demographic and geographic features"""
    features = df.copy()
    
    # Age bins (from EDA insights: 45-59 highest risk)
    age_bins = [18, 25, 30, 35, 40, 45, 50, 60, 100]
    age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-59", "60+"]
    features['age_bin'] = pd.cut(features['age'], bins=age_bins, labels=age_labels, right=False)
    
    # One-hot encode categorical features
    # Country (Germany is highest risk)
    features['country_Germany'] = (features['country'] == 'Germany').astype(int)
    features['country_France'] = (features['country'] == 'France').astype(int)
    features['country_Spain'] = (features['country'] == 'Spain').astype(int)
    
    # Gender (Female higher churn)
    features['gender_Female'] = (features['gender'] == 'Female').astype(int)
    features['gender_Male'] = (features['gender'] == 'Male').astype(int)
    
    # Age bins (45-59 highest risk)
    for age_label in age_labels:
        features[f'age_bin_{age_label.replace("-", "_")}'] = (features['age_bin'] == age_label).astype(int)
    
    return features

def create_financial_features(df):
    """Create financial and wealth-based features"""
    features = df.copy()
    
    # Balance tiers (from EDA: zero-balance is special segment)
    features['is_zero_balance'] = (features['balance'] == 0).astype(int)
    features['is_high_balance'] = (features['balance'] > 100000).astype(int)
    
    balance_bins = [0, 0.1, 25000, 50000, 100000, 150000, float('inf')]
    balance_labels = ['Zero', 'Low', 'Medium', 'High', 'Premium', 'Ultra']
    features['balance_tier'] = pd.cut(features['balance'], bins=balance_bins, labels=balance_labels, right=False)
    
    # One-hot encode balance tiers
    for tier in balance_labels:
        features[f'balance_tier_{tier}'] = (features['balance_tier'] == tier).astype(int)
    
    # Salary tiers
    salary_bins = [0, 50000, 75000, 100000, 125000, float('inf')]
    salary_labels = ['Low', 'Lower_Mid', 'Mid', 'Upper_Mid', 'High']
    features['salary_tier'] = pd.cut(features['estimated_salary'], bins=salary_bins, labels=salary_labels, right=False)
    
    # One-hot encode salary tiers
    for tier in salary_labels:
        features[f'salary_tier_{tier}'] = (features['salary_tier'] == tier).astype(int)
    
    # Credit score tiers (from existing data)
    for tier in ['Poor', 'Fair', 'Good', 'Excellent']:
        features[f'credit_score_tier_{tier}'] = (features['credit_score_tier'] == tier).astype(int)
    
    # Customer value tiers (from existing data)
    for tier in ['Low', 'Medium', 'High', 'Premium']:
        features[f'customer_value_tier_{tier}'] = (features['customer_value_tier'] == tier).astype(int)
    
    # Wealth ratios and derived features
    features['balance_to_salary_ratio_capped'] = np.clip(features['balance_to_salary_ratio'], 0, 5)  # Cap extreme outliers
    features['log_balance'] = np.log1p(features['balance'])  # Log transform for skewed distribution
    features['log_salary'] = np.log1p(features['estimated_salary'])
    features['wealth_index'] = features['log_balance'] + 0.1 * features['log_salary']  # Combined wealth measure
    
    return features

def create_engagement_features(df):
    """Create customer engagement and behavioral features"""
    features = df.copy()
    
    # Products (from EDA: 1→2 products huge churn reduction)
    features['has_single_product'] = (features['products_number'] == 1).astype(int)
    features['has_multiple_products'] = (features['products_number'] >= 2).astype(int)
    features['products_3_plus'] = (features['products_number'] >= 3).astype(int)
    
    # Products per year (adoption rate)
    features['products_per_year_capped'] = np.clip(features['products_per_year'], 0, 2)  # Cap outliers
    
    # Tenure segments (from EDA insights)
    tenure_bins = [0, 2, 4, 6, 8, 12]
    tenure_labels = ['New', 'Growing', 'Established', 'Mature', 'Veteran']
    features['tenure_tier'] = pd.cut(features['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
    
    # One-hot encode tenure tiers
    for tier in tenure_labels:
        features[f'tenure_tier_{tier}'] = (features['tenure_tier'] == tier).astype(int)
    
    # Engagement flags
    features['is_active_member'] = features['active_member']
    features['has_credit_card'] = features['credit_card']
    features['is_inactive_with_card'] = ((features['active_member'] == 0) & (features['credit_card'] == 1)).astype(int)
    features['is_active_without_card'] = ((features['active_member'] == 1) & (features['credit_card'] == 0)).astype(int)
    
    return features

def create_interaction_features(df):
    """Create interaction features based on EDA insights"""
    features = df.copy()
    
    # High-impact interactions from EDA
    
    # Country × Credit Score (Germany + Poor Credit = highest risk)
    features['germany_poor_credit'] = ((features['country'] == 'Germany') & 
                                      (features['credit_score_tier'] == 'Poor')).astype(int)
    features['germany_high_value'] = ((features['country'] == 'Germany') & 
                                     (features['customer_value_tier'] == 'High')).astype(int)
    
    # Age × Gender (older females highest risk)
    features['female_45_plus'] = ((features['gender'] == 'Female') & (features['age'] >= 45)).astype(int)
    features['male_young'] = ((features['gender'] == 'Male') & (features['age'] < 35)).astype(int)
    
    # Engagement × Products (active + multiple products = lowest risk)
    features['active_multiple_products'] = ((features['active_member'] == 1) & 
                                           (features['products_number'] >= 2)).astype(int)
    features['inactive_single_product'] = ((features['active_member'] == 0) & 
                                          (features['products_number'] == 1)).astype(int)
    
    # Zero balance interactions (special segment)
    features['zero_balance_active'] = ((features['balance'] == 0) & (features['active_member'] == 1)).astype(int)
    features['zero_balance_multiple_products'] = ((features['balance'] == 0) & 
                                                 (features['products_number'] >= 2)).astype(int)
    
    # Credit score × Customer value
    features['poor_credit_high_value'] = ((features['credit_score_tier'] == 'Poor') & 
                                         (features['customer_value_tier'].isin(['High', 'Premium']))).astype(int)
    features['excellent_credit_low_value'] = ((features['credit_score_tier'] == 'Excellent') & 
                                             (features['customer_value_tier'] == 'Low')).astype(int)
    
    return features

def create_risk_scores(df):
    """Create composite risk scores based on EDA insights"""
    features = df.copy()
    
    # Geographic risk score (Germany = 1, others = 0)
    features['geographic_risk_score'] = (features['country'] == 'Germany').astype(int)
    
    # Demographic risk score (age 45-59 + female)
    age_risk = ((features['age'] >= 45) & (features['age'] < 60)).astype(int)
    gender_risk = (features['gender'] == 'Female').astype(int)
    features['demographic_risk_score'] = age_risk + gender_risk  # 0-2 scale
    
    # Engagement risk score (inactive + single product)
    engagement_risk = (features['active_member'] == 0).astype(int)
    product_risk = (features['products_number'] == 1).astype(int)
    features['engagement_risk_score'] = engagement_risk + product_risk  # 0-2 scale
    
    # Credit risk score (Poor = 2, Fair = 1, Good/Excellent = 0)
    credit_risk_map = {'Poor': 2, 'Fair': 1, 'Good': 0, 'Excellent': 0}
    features['credit_risk_score'] = features['credit_score_tier'].map(credit_risk_map)
    
    # Composite risk score (weighted combination)
    features['composite_risk_score'] = (
        0.3 * features['geographic_risk_score'] +
        0.2 * features['demographic_risk_score'] / 2 +  # Normalize to 0-1
        0.3 * features['engagement_risk_score'] / 2 +   # Normalize to 0-1
        0.2 * features['credit_risk_score'] / 2         # Normalize to 0-1
    )
    
    return features

def scale_numeric_features(df):
    """Scale numeric features for ML"""
    features = df.copy()
    
    # Identify numeric features to scale
    numeric_features = [
        'credit_score', 'age', 'tenure', 'balance', 'estimated_salary',
        'balance_to_salary_ratio_capped', 'products_per_year_capped',
        'log_balance', 'log_salary', 'wealth_index',
        'composite_risk_score'
    ]
    
    # Scale features
    scaler = StandardScaler()
    for feature in numeric_features:
        if feature in features.columns:
            features[f'{feature}_scaled'] = scaler.fit_transform(features[[feature]])
    
    return features

def select_final_features(df):
    """Select final feature set for modeling"""
    
    # Target variable
    target = ['churn']
    
    # Identifier
    id_cols = ['customer_id']
    
    # Core demographic features (one-hot encoded)
    demographic_features = [col for col in df.columns if col.startswith(('country_', 'gender_', 'age_bin_'))]
    
    # Financial features (one-hot encoded + scaled)
    financial_features = [col for col in df.columns if col.startswith(('balance_tier_', 'salary_tier_', 
                                                                      'credit_score_tier_', 'customer_value_tier_'))]
    financial_features += ['log_balance_scaled', 'log_salary_scaled', 'wealth_index_scaled', 
                          'balance_to_salary_ratio_capped_scaled']
    
    # Engagement features
    engagement_features = [col for col in df.columns if col.startswith('tenure_tier_')]
    engagement_features += ['is_active_member', 'has_credit_card', 'has_single_product', 
                           'has_multiple_products', 'products_per_year_capped_scaled']
    
    # Interaction features
    interaction_features = [
        'germany_poor_credit', 'germany_high_value', 'female_45_plus', 'male_young',
        'active_multiple_products', 'inactive_single_product', 'zero_balance_active',
        'zero_balance_multiple_products', 'poor_credit_high_value', 'excellent_credit_low_value'
    ]
    
    # Risk scores
    risk_features = ['composite_risk_score_scaled', 'geographic_risk_score', 
                    'demographic_risk_score', 'engagement_risk_score', 'credit_risk_score']
    
    # Special flags
    flag_features = ['is_zero_balance', 'is_high_balance', 'products_3_plus']
    
    # Combine all features
    final_features = (id_cols + target + demographic_features + financial_features + 
                     engagement_features + interaction_features + risk_features + flag_features)
    
    # Select only features that exist in dataframe
    final_features = [col for col in final_features if col in df.columns]
    
    return df[final_features]

def analyze_features(df):
    """Analyze feature importance and correlations"""
    
    # Feature correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['churn'].abs().sort_values(ascending=False)
    
    # Remove target self-correlation
    feature_importance = correlations.drop('churn').head(20)
    
    # Feature correlation matrix for multicollinearity check
    feature_corr_matrix = df[numeric_cols].corr()
    
    return feature_importance, feature_corr_matrix

def main():
    ensure_out_dir()
    
    print("🔧 BANKING FEATURE ENGINEERING")
    print("="*50)
    
    # Load data
    print("📥 Loading banking data...")
    bank = load_bank()
    print(f"   • Loaded {len(bank):,} customers")
    
    # Create features step by step
    print("\n🏗️ Creating features...")
    
    print("   • Demographic features...")
    bank = create_demographic_features(bank)
    
    print("   • Financial features...")
    bank = create_financial_features(bank)
    
    print("   • Engagement features...")
    bank = create_engagement_features(bank)
    
    print("   • Interaction features...")
    bank = create_interaction_features(bank)
    
    print("   • Risk scores...")
    bank = create_risk_scores(bank)
    
    print("   • Scaling numeric features...")
    bank = scale_numeric_features(bank)
    
    print("   • Selecting final feature set...")
    final_features = select_final_features(bank)
    
    print(f"\n📊 Final dataset shape: {final_features.shape}")
    print(f"   • Features: {final_features.shape[1] - 2}")  # Exclude customer_id and churn
    print(f"   • Samples: {final_features.shape[0]:,}")
    
    # Analyze features
    print("\n🔍 Analyzing features...")
    feature_importance, feature_corr_matrix = analyze_features(final_features)
    
    print("\n🎯 Top 10 features by correlation with churn:")
    for i, (feature, corr) in enumerate(feature_importance.head(10).items(), 1):
        print(f"   {i:2d}. {feature:<35} {corr:.4f}")
    
    # Save outputs
    print(f"\n💾 Saving outputs to {OUT_DIR.resolve()}...")
    
    # Save final features
    final_features.to_csv(OUT_DIR / 'banking_features.csv', index=False)
    
    # Save feature analysis
    feature_importance.to_csv(OUT_DIR / 'feature_importance_eda.csv')
    feature_corr_matrix.to_csv(OUT_DIR / 'feature_correlation_matrix.csv')
    
    # Summary statistics
    summary = {
        'total_customers': len(final_features),
        'total_features': final_features.shape[1] - 2,
        'churn_rate': final_features['churn'].mean(),
        'top_feature': feature_importance.index[0],
        'top_correlation': feature_importance.iloc[0]
    }
    
    print("\n✅ Feature engineering completed!")
    print(f"   • Model-ready dataset: features/banking_features.csv")
    print(f"   • Feature importance: features/feature_importance_eda.csv")
    print(f"   • Correlation matrix: features/feature_correlation_matrix.csv")
    print(f"   • Ready for model training with {summary['total_features']} features!")
    
    return summary

if __name__ == "__main__":
    summary = main()
