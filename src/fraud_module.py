import pandas as pd 
import numpy as np 
 
class DataProcessor: 
    def clean_data(self, df): 
        \"\"\"Clean data with error handling\"\"\" 
        try: 
            df_clean = df.copy() 
            return df_clean 
        except Exception as e: 
            print(f\"Error: {e}\") 
            return df 
