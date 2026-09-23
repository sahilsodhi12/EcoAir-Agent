import pandas as pd
import numpy as np

# ============================================================
# ECOAIR AGENT - DATA CLEANING
# ============================================================

# 1. Load the original dataset
df = pd.read_csv(
    "data/raw/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

print("=" * 60)
print("ECOAIR AGENT - DATA CLEANING")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)

# ============================================================
# 2. Remove completely empty columns
# ============================================================

df = df.dropna(axis=1, how="all")

print("\nAfter removing empty columns:")
print(df.shape)

# ============================================================
# 3. Convert -200 to NaN
# ============================================================

print("\nConverting -200 values into missing values...")

df = df.replace(-200, np.nan)

# ============================================================
# 4. Check missing values
# ============================================================

print("\nMissing values in each column:")
print(df.isna().sum())

# ============================================================
# 5. Missing-value percentage
# ============================================================

print("\nMissing-value percentage:")
missing_percentage = (df.isna().sum() / len(df)) * 100

print(missing_percentage.round(2))

# ============================================================
# 6. Convert Date and Time into a single datetime column
# ============================================================

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
    dayfirst=True,
    errors="coerce"
)

print("\nDatetime conversion completed.")

# ============================================================
# 7. Remove original Date and Time columns
# ============================================================

df = df.drop(columns=["Date", "Time"])

# ============================================================
# 8. Fill missing numerical values with median
# ============================================================

numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing numerical values filled using median.")

# ============================================================
# 9. Check if missing values remain
# ============================================================

print("\nRemaining missing values:")
print(df.isna().sum().sum())

# ============================================================
# 10. Save cleaned dataset
# ============================================================

df.to_csv(
    "data/processed_air_quality.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# ============================================================
# 11. Final information
# ============================================================

print("\nFinal dataset shape:")
print(df.shape)

print("\nFirst 5 cleaned rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)