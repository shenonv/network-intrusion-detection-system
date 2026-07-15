from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")

csv_files = list(RAW_DATA_DIR.glob("*.csv"))

if len(csv_files) == 0:
    print("No CSV files found in data/raw/")
    print("Please put one CSE-CIC-IDS2018 CSV file inside data/raw/")
    exit()

csv_path = csv_files[0]
print(f"Reading file: {csv_path}")

# low_memory=False reduces mixed type warning
df = pd.read_csv(csv_path, low_memory=False)

print("\nBefore cleaning:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove repeated header rows
df = df[df["Label"] != "Label"]

print("\nAfter removing repeated header rows:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["Label"].value_counts())