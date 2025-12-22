
"""
Fraud Detection Utilities Module
Contains helper functions for fraud detection pipeline
"""

import pandas as pd
import numpy as np

def ip_to_int(ip_address):
    """
    Convert IP address to integer for geolocation lookup
    """
    try:
        if isinstance(ip_address, float):
            ip_int = int(ip_address)
        else:
            parts = str(ip_address).split('.')
            if len(parts) == 4:
                ip_int = (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
            else:
                ip_int = int(float(ip_address))
        return ip_int
    except Exception as e:
        print(f"Error converting IP {ip_address}: {e}")
        return None

def create_time_features(df, signup_col='signup_time', purchase_col='purchase_time'):
    """
    Create time-based features from signup and purchase timestamps
    """
    df = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[signup_col]):
        df[signup_col] = pd.to_datetime(df[signup_col])
    if not pd.api.types.is_datetime64_any_dtype(df[purchase_col]):
        df[purchase_col] = pd.to_datetime(df[purchase_col])
    
    # Time difference in hours
    df['time_since_signup_hours'] = (df[purchase_col] - df[signup_col]).dt.total_seconds() / 3600
    
    # Time-based flags
    df['is_immediate_purchase'] = (df['time_since_signup_hours'] <= 1).astype(int)
    df['hour_of_day'] = df[purchase_col].dt.hour
    df['day_of_week'] = df[purchase_col].dt.dayofweek
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def print_success(message):
    """Print success message with simple formatting"""
    print(f"[SUCCESS] {message}")

def print_error(message):
    """Print error message with simple formatting"""
    print(f"[ERROR] {message}")
