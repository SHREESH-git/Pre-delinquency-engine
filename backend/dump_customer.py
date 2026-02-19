
import pandas as pd
import sys

# Path to the dataset
# "d:\Games\Frontend\content_backup (2)\content_backup\content\financial_stress_full_bank_grade_dataset.csv"
csv_path = r"d:\Games\Frontend\content_backup (2)\content_backup\content\financial_stress_full_bank_grade_dataset.csv"


try:
    df = pd.read_csv(csv_path)
    cust = df[df['customer_id'] == 'CUST0000001'].sort_values('month')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    with open("cust_dump.txt", "w") as f:
        f.write(cust.to_string())
    
except Exception as e:
    print(f"Error: {e}")
