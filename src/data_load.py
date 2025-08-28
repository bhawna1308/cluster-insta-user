
import pandas as pd

# -------- Load Data --------
file_path = r"C:\Users\bhumi\Desktop\AI capstone project\my capstone 1\project\data\sample.csv" 
df = pd.read_csv(file_path)
print("✅ Data Loaded")

# -------- Dataset Info --------
print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 First 5 rows:")
print(df.head())

# -------- Check for Null Values --------
print("\n📌 Null Values in Each Column:")
print(df.isnull().sum())

# -------- Check for Duplicates --------
duplicates = df.duplicated().sum()
print(f"\n📌 Number of duplicate rows: {duplicates}")

# -------- Handle Missing Values (Optional) --------
# Example: Fill numeric with mean, categorical with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print("✅ Missing values handled")

# -------- Drop Duplicates --------
df.drop_duplicates(inplace=True)
print("✅ Duplicates removed")


# -------- Final Preprocessed Data --------
print("\n📌 Final Preprocessed Data (first 5 rows):")
print(df.head())
