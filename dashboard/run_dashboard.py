#!/usr/bin/env python3
"""
Quick launcher for Risk & Retention Intelligence Engine Dashboard
Run this script to start the dashboard locally
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required = ['flask', 'pandas', 'numpy', 'sqlite3']
    missing = []
    
    for package in required:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_requirements():
    """Install missing requirements"""
    print("🔧 Installing required packages...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    print("✅ Packages installed successfully!")

def main():
    print("🏦 Risk & Retention Intelligence Engine Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: Please run this script from the dashboard directory")
        print("💡 Try: cd dashboard && python run_dashboard.py")
        sys.exit(1)
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        response = input("📦 Install missing packages? (y/n): ").lower().strip()
        if response == 'y':
            try:
                install_requirements()
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages. Please run manually:")
                print("   pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("❌ Cannot continue without required packages")
            sys.exit(1)
    
    # Check if database exists
    db_path = Path("../database/banking_insights.db")
    if not db_path.exists():
        print(f"⚠️  Database not found: {db_path}")
        print("💡 Dashboard will work with simulated predictions")
    else:
        print(f"✅ Database found: {db_path}")
    
    # Check if models exist
    models_dir = Path("../models")
    banking_model = models_dir / "banking_churn_model_final.pkl"
    fraud_model = models_dir / "fraud_detection_model_ultimate.pkl"
    
    if banking_model.exists():
        print(f"✅ Banking model found: {banking_model}")
    else:
        print(f"⚠️  Banking model not found: {banking_model}")
        print("💡 Will use business rules fallback")
    
    if fraud_model.exists():
        print(f"✅ Fraud model found: {fraud_model}")
    else:
        print(f"⚠️  Fraud model not found: {fraud_model}")
        print("💡 Will use simulation fallback")
    
    print("\n🚀 Starting dashboard server...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 Features:")
    print("   • Executive Overview with KPIs")
    print("   • Interactive Data Insights")
    print("   • ML Model Performance")
    print("   • Live Predictions (Banking Churn + Fraud)")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("💡 Try running directly: python app.py")

if __name__ == "__main__":
    main()
