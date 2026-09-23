import pandas as pd
import joblib


# ============================================================
# ECOAIR AGENT - FIND REAL EXAMPLES
# ============================================================

print("=" * 70)
print("ECOAIR AGENT - FIND NORMAL, ELEVATED AND HIGH EXAMPLES")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load("ecoair_random_forest.pkl")

print("\nRandom Forest model loaded successfully.")


# ============================================================
# 2. LOAD SENSOR DATA
# ============================================================

X = pd.read_csv("data/X_features.csv")

print("Sensor data loaded.")
print("Total observations:", len(X))


# ============================================================
# 3. DECISION THRESHOLDS
# ============================================================

NORMAL_THRESHOLD = 1.1
HIGH_THRESHOLD = 2.9


# ============================================================
# 4. FIND EXAMPLES
# ============================================================

normal_example = None
elevated_example = None
high_example = None


# ============================================================
# 5. PREDICT FOR ALL OBSERVATIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X)


# ============================================================
# 6. FIND ONE EXAMPLE FOR EACH STATE
# ============================================================

for i, prediction in enumerate(predictions):

    # NORMAL
    if prediction <= NORMAL_THRESHOLD and normal_example is None:

        normal_example = (i, prediction)

    # ELEVATED
    elif (
        prediction > NORMAL_THRESHOLD
        and prediction <= HIGH_THRESHOLD
        and elevated_example is None
    ):

        elevated_example = (i, prediction)

    # HIGH
    elif prediction > HIGH_THRESHOLD and high_example is None:

        high_example = (i, prediction)

    # Stop once all three have been found
    if (
        normal_example is not None
        and elevated_example is not None
        and high_example is not None
    ):
        break


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REAL AGENT EXAMPLES")
print("=" * 70)


# NORMAL
if normal_example:

    index, prediction = normal_example

    print("\nNORMAL")
    print("-" * 70)
    print("Observation:", index)
    print(f"Predicted CO: {prediction:.2f}")
    print("Decision: MONITOR")


# ELEVATED
if elevated_example:

    index, prediction = elevated_example

    print("\nELEVATED")
    print("-" * 70)
    print("Observation:", index)
    print(f"Predicted CO: {prediction:.2f}")
    print("Decision: WARNING")


# HIGH
if high_example:

    index, prediction = high_example

    print("\nHIGH")
    print("-" * 70)
    print("Observation:", index)
    print(f"Predicted CO: {prediction:.2f}")
    print("Decision: ALERT")


# ============================================================
# 8. SHOW SENSOR VALUES
# ============================================================

print("\n" + "=" * 70)
print("SENSOR VALUES FOR THE EXAMPLES")
print("=" * 70)


examples = [
    ("NORMAL", normal_example),
    ("ELEVATED", elevated_example),
    ("HIGH", high_example)
]


for state, example in examples:

    if example is not None:

        index, prediction = example

        print("\n" + "-" * 70)
        print(state)
        print("-" * 70)

        print(X.iloc[index].to_string())

        print(f"\nPredicted CO: {prediction:.2f}")


print("\n" + "=" * 70)
print("SEARCH COMPLETED")
print("=" * 70)