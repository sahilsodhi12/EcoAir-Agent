import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ECOAIR AGENT - DATA ANALYSIS
# ============================================================

# Load prepared model data
X = pd.read_csv("data/X_features.csv")
y = pd.read_csv("data/y_target.csv").squeeze()

# Combine features and target
data = X.copy()
data["CO_Concentration"] = y

# ============================================================
# CORRELATION MATRIX
# ============================================================

correlation = data.corr()

print("=" * 60)
print("CORRELATION WITH CO CONCENTRATION")
print("=" * 60)

print(
    correlation["CO_Concentration"]
    .sort_values(ascending=False)
)

# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Between Environmental Variables"
)

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.show()

print("\nCorrelation heatmap saved successfully.")