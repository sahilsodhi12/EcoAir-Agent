import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# ECOAIR AGENT - RANDOM FOREST MODEL
# ============================================================

print("=" * 65)
print("ECOAIR AGENT - RANDOM FOREST REGRESSOR")
print("=" * 65)


# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================

X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_target.csv").squeeze()

print("\nData loaded successfully.")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 2. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nData split completed.")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 3. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nRandom Forest model created.")

print("Number of trees:", 100)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nPredictions generated.")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("MODEL RESULTS")
print("=" * 65)

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("\n" + "=" * 65)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("=" * 65)

print(importance)


# ============================================================
# 9. SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "ecoair_random_forest.pkl"
)

print("\nModel saved as:")
print("ecoair_random_forest.pkl")

print("\nMODEL TRAINING COMPLETED")
print("=" * 65)