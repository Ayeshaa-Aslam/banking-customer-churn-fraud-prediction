"""
Banking Risk & Retention Intelligence Engine
01_banking_database_setup.py - Banking Database Creation and Data Import

This script creates the SQLite database for banking-focused analysis.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankingDatabaseSetup:
    def __init__(self, db_path="database/banking_insights.db"):
        """Initialize banking database setup"""
        self.db_path = db_path
        self.data_dir = Path("data")
        
        # Create database directory if it doesn't exist
        Path(db_path).parent.mkdir(exist_ok=True)
        
    def create_banking_schema(self):
        """Create database tables with banking-focused schema"""
        logger.info("Creating banking database schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create fraud_data table (unchanged)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_data (
            id INTEGER PRIMARY KEY,
            Time REAL,
            V1 REAL, V2 REAL, V3 REAL, V4 REAL, V5 REAL,
            V6 REAL, V7 REAL, V8 REAL, V9 REAL, V10 REAL,
            V11 REAL, V12 REAL, V13 REAL, V14 REAL, V15 REAL,
            V16 REAL, V17 REAL, V18 REAL, V19 REAL, V20 REAL,
            V21 REAL, V22 REAL, V23 REAL, V24 REAL, V25 REAL,
            V26 REAL, V27 REAL, V28 REAL,
            Amount REAL,
            Class INTEGER,
            customer_id INTEGER  -- For linking
        )
        """)
        
        # Create banking_customers table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS banking_customers (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER UNIQUE,
            credit_score INTEGER,
            country TEXT,
            gender TEXT,
            age INTEGER,
            tenure INTEGER,
            balance REAL,
            products_number INTEGER,
            credit_card INTEGER,
            active_member INTEGER,
            estimated_salary REAL,
            churn INTEGER,
            -- Derived fields for analysis
            credit_score_tier TEXT,
            customer_value_tier TEXT,
            balance_to_salary_ratio REAL,
            products_per_year REAL,
            is_high_value INTEGER,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create customer_links table (for integrated analysis)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_links (
            id INTEGER PRIMARY KEY,
            banking_customer_id INTEGER,
            fraud_customer_id INTEGER,
            link_strength REAL,  -- How strong the connection is
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (banking_customer_id) REFERENCES banking_customers(customer_id)
        )
        """)
        
        # Create banking_insights table (for analysis results)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS banking_insights (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            churn_risk_score REAL,
            fraud_risk_score REAL,
            combined_risk_score REAL,
            customer_lifetime_value REAL,
            retention_priority INTEGER,
            recommended_action TEXT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES banking_customers(customer_id)
        )
        """)
        
        # Create audit log table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            operation TEXT,
            table_name TEXT,
            record_count INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Banking database schema created successfully!")
        
    def import_fraud_data(self):
        """Import credit card fraud dataset"""
        logger.info("Importing fraud dataset...")
        
        fraud_file = self.data_dir / "creditcard.csv"
        if not fraud_file.exists():
            raise FileNotFoundError(f"Fraud dataset not found: {fraud_file}")
            
        df_fraud = pd.read_csv(fraud_file)
        logger.info(f"Loaded fraud data: {len(df_fraud)} transactions")
        
        # Add customer_id for linking (same as before)
        np.random.seed(42)
        df_fraud['customer_id'] = np.random.randint(1, 50000, len(df_fraud))
        
        # Import to database
        conn = sqlite3.connect(self.db_path)
        df_fraud.to_sql('fraud_data', conn, if_exists='replace', index=False)
        
        # Log the import
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO audit_log (operation, table_name, record_count, notes)
        VALUES (?, ?, ?, ?)
        """, ('IMPORT', 'fraud_data', len(df_fraud), 'Credit card fraud dataset import'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Fraud data imported: {len(df_fraud)} records")
        return len(df_fraud)
        
    def import_banking_data(self):
        """Import banking customer churn dataset"""
        logger.info("Importing banking customer dataset...")
        
        banking_file = self.data_dir / "Bank Customer Churn Prediction.csv"
        if not banking_file.exists():
            raise FileNotFoundError(f"Banking dataset not found: {banking_file}")
            
        df_banking = pd.read_csv(banking_file)
        logger.info(f"Loaded banking data: {len(df_banking)} customers")
        
        # Clean and enhance the data
        df_banking = self.enhance_banking_data(df_banking)
        
        # Import to database
        conn = sqlite3.connect(self.db_path)
        df_banking.to_sql('banking_customers', conn, if_exists='replace', index=False)
        
        # Log the import
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO audit_log (operation, table_name, record_count, notes)
        VALUES (?, ?, ?, ?)
        """, ('IMPORT', 'banking_customers', len(df_banking), 'Banking customer churn dataset import'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Banking data imported: {len(df_banking)} records")
        return len(df_banking)
        
    def enhance_banking_data(self, df):
        """Enhance banking data with derived features"""
        logger.info("Enhancing banking data with derived features...")
        
        # Credit score tiers
        def get_credit_tier(score):
            if score < 580:
                return 'Poor'
            elif score < 670:
                return 'Fair'  
            elif score < 740:
                return 'Good'
            else:
                return 'Excellent'
                
        df['credit_score_tier'] = df['credit_score'].apply(get_credit_tier)
        
        # Customer value tiers based on balance and salary
        df['customer_value'] = df['balance'] + (df['estimated_salary'] * 0.1)  # Simple formula
        
        def get_value_tier(value):
            if value < 25000:
                return 'Low'
            elif value < 75000:
                return 'Medium'
            elif value < 150000:
                return 'High'
            else:
                return 'Premium'
                
        df['customer_value_tier'] = df['customer_value'].apply(get_value_tier)
        
        # Balance to salary ratio
        df['balance_to_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)  # +1 to avoid division by zero
        
        # Products per year (product adoption rate)
        df['products_per_year'] = df['products_number'] / (df['tenure'] + 1)  # +1 to avoid division by zero
        
        # High value customer flag (balance > 100K or salary > 100K)
        df['is_high_value'] = ((df['balance'] > 100000) | (df['estimated_salary'] > 100000)).astype(int)
        
        # Drop the temporary customer_value column
        df = df.drop('customer_value', axis=1)
        
        logger.info("Banking data enhancement completed!")
        return df
        
    def create_customer_links(self):
        """Create links between banking customers and fraud data"""
        logger.info("Creating customer links...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get customer IDs from both datasets
        banking_customers = pd.read_sql("SELECT customer_id FROM banking_customers", conn)
        fraud_customers = pd.read_sql("SELECT DISTINCT customer_id FROM fraud_data", conn)
        
        # Create realistic links (some customers appear in both datasets)
        np.random.seed(42)
        n_links = min(2000, len(banking_customers))  # Link up to 2000 customers
        
        # Sample banking customers
        linked_banking = banking_customers.sample(n_links)['customer_id'].values
        
        # Sample fraud customers (with replacement to allow multiple transactions per customer)
        linked_fraud = np.random.choice(fraud_customers['customer_id'].values, n_links)
        
        # Create link strength (0.1 to 1.0)
        link_strengths = np.random.uniform(0.1, 1.0, n_links)
        
        # Create links dataframe
        links_df = pd.DataFrame({
            'banking_customer_id': linked_banking,
            'fraud_customer_id': linked_fraud,
            'link_strength': link_strengths
        })
        
        # Remove duplicates
        links_df = links_df.drop_duplicates(subset=['banking_customer_id', 'fraud_customer_id'])
        
        links_df.to_sql('customer_links', conn, if_exists='replace', index=False)
        
        # Log the operation
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO audit_log (operation, table_name, record_count, notes)
        VALUES (?, ?, ?, ?)
        """, ('CREATE', 'customer_links', len(links_df), 'Banking-fraud customer links created'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Customer links created: {len(links_df)} links")
        return len(links_df)
        
    def verify_banking_import(self):
        """Verify banking data import and show comprehensive statistics"""
        logger.info("Verifying banking data import...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Banking customer statistics
        banking_stats = pd.read_sql("""
        SELECT 
            COUNT(*) as total_customers,
            SUM(churn) as churned_customers,
            AVG(credit_score) as avg_credit_score,
            AVG(balance) as avg_balance,
            AVG(estimated_salary) as avg_salary,
            COUNT(CASE WHEN is_high_value = 1 THEN 1 END) as high_value_customers
        FROM banking_customers
        """, conn).iloc[0]
        
        # Credit score distribution
        credit_dist = pd.read_sql("""
        SELECT credit_score_tier, COUNT(*) as count
        FROM banking_customers 
        GROUP BY credit_score_tier
        ORDER BY 
            CASE credit_score_tier
                WHEN 'Poor' THEN 1
                WHEN 'Fair' THEN 2
                WHEN 'Good' THEN 3
                WHEN 'Excellent' THEN 4
            END
        """, conn)
        
        # Geographic distribution
        geo_dist = pd.read_sql("""
        SELECT country, COUNT(*) as count, 
               AVG(churn) as churn_rate
        FROM banking_customers 
        GROUP BY country
        """, conn)
        
        # Churn by customer value
        value_churn = pd.read_sql("""
        SELECT customer_value_tier, COUNT(*) as count,
               AVG(churn) as churn_rate
        FROM banking_customers
        GROUP BY customer_value_tier
        ORDER BY 
            CASE customer_value_tier
                WHEN 'Low' THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'High' THEN 3
                WHEN 'Premium' THEN 4
            END
        """, conn)
        
        # Fraud data stats
        fraud_stats = pd.read_sql("""
        SELECT 
            COUNT(*) as total_transactions,
            SUM(Class) as fraud_transactions
        FROM fraud_data
        """, conn).iloc[0]
        
        # Customer links
        links_count = pd.read_sql("SELECT COUNT(*) as count FROM customer_links", conn).iloc[0]['count']
        
        conn.close()
        
        # Display comprehensive summary
        print("\n" + "="*80)
        print("BANKING RISK & RETENTION INTELLIGENCE ENGINE - DATABASE SUMMARY")
        print("="*80)
        
        print("🏦 BANKING CUSTOMER ANALYSIS:")
        print(f"   • Total Customers: {banking_stats['total_customers']:,}")
        print(f"   • Churned Customers: {banking_stats['churned_customers']:,} ({banking_stats['churned_customers']/banking_stats['total_customers']*100:.1f}%)")
        print(f"   • High-Value Customers: {banking_stats['high_value_customers']:,} ({banking_stats['high_value_customers']/banking_stats['total_customers']*100:.1f}%)")
        print(f"   • Average Credit Score: {banking_stats['avg_credit_score']:.0f}")
        print(f"   • Average Balance: ${banking_stats['avg_balance']:,.0f}")
        print(f"   • Average Salary: ${banking_stats['avg_salary']:,.0f}")
        
        print(f"\n💳 CREDIT SCORE DISTRIBUTION:")
        for _, row in credit_dist.iterrows():
            pct = row['count'] / banking_stats['total_customers'] * 100
            print(f"   • {row['credit_score_tier']}: {row['count']:,} ({pct:.1f}%)")
            
        print(f"\n🌍 GEOGRAPHIC ANALYSIS:")
        for _, row in geo_dist.iterrows():
            print(f"   • {row['country']}: {row['count']:,} customers, {row['churn_rate']*100:.1f}% churn rate")
            
        print(f"\n💰 CUSTOMER VALUE TIERS:")
        for _, row in value_churn.iterrows():
            print(f"   • {row['customer_value_tier']} Value: {row['count']:,} customers, {row['churn_rate']*100:.1f}% churn rate")
        
        print(f"\n🚨 FRAUD DATA:")
        fraud_rate = fraud_stats['fraud_transactions'] / fraud_stats['total_transactions']
        print(f"   • Total Transactions: {fraud_stats['total_transactions']:,}")
        print(f"   • Fraudulent: {fraud_stats['fraud_transactions']:,} ({fraud_rate*100:.2f}%)")
        
        print(f"\n🔗 CUSTOMER INTEGRATION:")
        print(f"   • Linked Customers: {links_count:,}")
        
        print(f"\n✅ Database Location: {self.db_path}")
        print("="*80)
        
        return {
            'banking_stats': banking_stats.to_dict(),
            'credit_distribution': credit_dist.to_dict('records'),
            'geographic_distribution': geo_dist.to_dict('records'),
            'value_tier_analysis': value_churn.to_dict('records'),
            'fraud_stats': fraud_stats.to_dict(),
            'linked_customers': links_count
        }

def main():
    """Main function to set up the banking database"""
    logger.info("Starting banking database setup...")
    
    try:
        # Initialize database setup
        db_setup = BankingDatabaseSetup()
        
        # Create schema
        db_setup.create_banking_schema()
        
        # Import data
        fraud_records = db_setup.import_fraud_data()
        banking_records = db_setup.import_banking_data()
        
        # Create customer links
        linked_customers = db_setup.create_customer_links()
        
        # Verify import
        summary = db_setup.verify_banking_import()
        
        logger.info("Banking database setup completed successfully!")
        
        return summary
        
    except Exception as e:
        logger.error(f"Banking database setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    summary = main()
