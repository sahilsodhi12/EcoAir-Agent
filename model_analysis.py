import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# ECOAIR AGENT - MODEL ANALYSIS
# ============================================================

print("=" * 65)
print("ECOAIR AGENT - MODEL ANALYSIS")
print("=" * 65)


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_target.csv").squeeze()


# ------------------------------------------------------------
# 2. Same train/test split as before
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# 3. Train Random Forest
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 5. Calculate metrics
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nMODEL PERFORMANCE")
print("-" * 65)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 6. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual CO Concentration")
plt.ylabel("Predicted CO Concentration")

plt.title(
    "Actual vs Predicted CO Concentration"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. FEATURE IMPORTANCE
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
print("-" * 65)

print(importance)


# ============================================================
# 8. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


print("\nGraphs saved successfully.")

print("\n" + "=" * 65)
print("MODEL ANALYSIS COMPLETED")
print("=" * 65)