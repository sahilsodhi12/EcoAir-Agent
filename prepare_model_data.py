import pandas as pd
import numpy as np

# ============================================================
# ECOAIR AGENT - PREPARE MODEL DATA
# ============================================================

# 1. Load original UCI dataset
df = pd.read_csv(
    "data/raw/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

# 2. Remove completely empty columns
df = df.dropna(axis=1, how="all")

# 3. Convert -200 into missing values
df = df.replace(-200, np.nan)

# ============================================================
# 4. Create datetime
# ============================================================

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
    dayfirst=True,
    errors="coerce"
)

# Remove rows with invalid date/time
df = df.dropna(subset=["Datetime"])

# ============================================================
# 5. Original features
# ============================================================

original_features = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

# Target
original_target = "CO(GT)"

# ============================================================
# 6. Create X and y
# ============================================================

X = df[original_features].copy()
y = df[original_target].copy()

# ============================================================
# 7. Remove rows where target is missing
# ============================================================

valid_target = y.notna()

X = X[valid_target]
y = y[valid_target]

# ============================================================
# 8. Fill missing feature values with median
# ============================================================

for column in X.columns:
    X[column] = X[column].fillna(X[column].median())

# ============================================================
# 9. Rename columns to informative names
# ============================================================

feature_name_mapping = {

    "PT08.S1(CO)": "CO_Sensor_1",

    "C6H6(GT)": "Benzene_Concentration",

    "PT08.S2(NMHC)": "NMHC_Sensor",

    "NOx(GT)": "NOx_Concentration",

    "PT08.S3(NOx)": "NOx_Sensor",

    "NO2(GT)": "NO2_Concentration",

    "PT08.S4(NO2)": "NO2_Sensor",

    "PT08.S5(O3)": "O3_Sensor",

    "T": "Temperature_C",

    "RH": "Relative_Humidity",

    "AH": "Absolute_Humidity"
}

X = X.rename(columns=feature_name_mapping)

# Rename target
y = y.rename("CO_Concentration")

# ============================================================
# 10. Display information
# ============================================================

print("=" * 65)
print("ECOAIR AGENT - MODEL DATA")
print("=" * 65)

print("\nNumber of observations:", len(X))

print("\nInput features:")
for feature in X.columns:
    print("-", feature)

print("\nTarget:")
print("-", y.name)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nMissing values in X:")
print(X.isna().sum().sum())

print("\nMissing values in y:")
print(y.isna().sum())

print("\nFirst 5 input records:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())

# ============================================================
# 11. Save processed data
# ============================================================

X.to_csv(
    "data/X_features.csv",
    index=False
)

y.to_csv(
    "data/y_target.csv",
    index=False
)

print("\nProcessed files saved successfully.")

print("\n" + "=" * 65)
print("MODEL DATA PREPARATION COMPLETED")
print("=" * 65)