import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


csv_files = list(RAW_DATA_DIR.glob("*.csv"))

if len(csv_files) == 0:
    print("No CSV files found in data/raw/")
    exit()

print(f"Found {len(csv_files)} CSV file(s).")

dataframes = []

for csv_path in csv_files:
    print(f"Reading: {csv_path}")

    df_temp = pd.read_csv(csv_path, low_memory=False)

    df_temp.columns = df_temp.columns.str.strip()

    if "Label" in df_temp.columns:
        df_temp = df_temp[df_temp["Label"] != "Label"]

    dataframes.append(df_temp)

df = pd.concat(dataframes, ignore_index=True)

print("\nDataset loaded.")
print(f"Rows before cleaning: {df.shape[0]}")
print(f"Columns before cleaning: {df.shape[1]}")

df.columns = df.columns.str.strip()

if "Label" not in df.columns:
    print("Error: Label column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

df["Label"] = df["Label"].astype(str).str.strip()

df["Label"] = df["Label"].replace({
    "Infilteration":"Infiltration"
})

if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

df = df[df["Label"].notna()]
df = df[df["Label"] !=""]

X = df.drop(columns=["Label"])
y = df["Label"]

print("\nAfter basic label cleaning:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nCurrent label counts:")
print(y.value_counts())

print("\nFeature columns:")
print(X.columns.tolist())

print(f"\nNumber of features before numeric cleaning: {X.shape[1]}")

# -----------------------------
# Numeric cleaning
# -----------------------------
print("\nConverting feature columns to numeric...")

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

missing_values = X.isna().sum().sum()
infinity_values = np.isinf(X.to_numpy()).sum()

print(f"Missing values after numeric conversion: {missing_values}")
print(f"Infinity values before cleaning: {infinity_values}")

X = X.replace([np.inf, -np.inf], np.nan)

clean_df = X.copy()  
clean_df["Label"] = y.values

rows_before_drop = clean_df.shape[0]

clean_df = clean_df.dropna()

rows_after_drop = clean_df.shape[0]
rows_removed = rows_before_drop - rows_after_drop

X = clean_df.drop(columns=["Label"])
y = clean_df["Label"]

print(f"Rows removed because of NaN/Infinity: {rows_removed}")

print("\nAfter numeric cleaning:")
print(f"Rows: {X.shape[0]}")
print(f"Features: {X.shape[1]}")

print("\nFinal label counts after numeric cleaning:")
print(y.value_counts())

# -----------------------------
# Encode labels
# -----------------------------

print("\nEncoding labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEncoded label mapping:")

for class_id, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {class_id}")

print("\nFirst 10 encoded labels:")
print(y_encoded[:10])

# -----------------------------
# Train/test split
# -----------------------------
print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain/test split completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nTraining label distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting label distribution:")
print(pd.Series(y_test).value_counts())

# -----------------------------
# Feature scaling
# -----------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed.")
print("\nScaled data shapes:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

print("\nExample before scaling:")
print(X_train.iloc[0].values[:5])

print("\nExample after scaling:")
print(X_train_scaled[0][:5])

# -----------------------------
# Save processed arrays
# -----------------------------

print("\nSaving processed data files...")

np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train_scaled)
np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test_scaled)
np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

feature_names = X.columns.tolist()

with open(MODELS_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=4)

print("\nFiles saved successfully:")
print("data/processed/X_train.npy")
print("data/processed/X_test.npy")
print("data/processed/y_train.npy")
print("data/processed/y_test.npy")
print("models/scaler.pkl")
print("models/label_encoder.pkl")
print("models/feature_names.json")

print("\nPreprocessing completed successfully.")