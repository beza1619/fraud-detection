# src/fraud_pipeline.py
import pandas as pd
import numpy as np

class FraudDetector:
    def __init__(self):
        self.model = None
        
    def load_data(self, path):
        """Load and validate data"""
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def clean_data(self, df):
        """Clean data with error handling"""
        df_clean = df.copy()
        # Add cleaning logic
        return df_clean