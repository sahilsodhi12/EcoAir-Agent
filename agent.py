import pandas as pd
import joblib


# ============================================================
# ECOAIR AGENT - SEQUENTIAL MONITORING
# ============================================================

print("=" * 70)
print("ECOAIR AGENT - SEQUENTIAL ENVIRONMENT MONITORING")
print("=" * 70)


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

model = joblib.load("ecoair_random_forest.pkl")

print("\nRandom Forest model loaded successfully.")


# ============================================================
# 2. DECISION THRESHOLDS
# ============================================================

NORMAL_THRESHOLD = 1.1
HIGH_THRESHOLD = 2.9


# ============================================================
# 3. AGENT DECISION FUNCTION
# ============================================================

def make_decision(predicted_co):

    if predicted_co <= NORMAL_THRESHOLD:

        state = "NORMAL"
        action = "MONITOR"

    elif predicted_co <= HIGH_THRESHOLD:

        state = "ELEVATED"
        action = "WARNING"

    else:

        state = "HIGH"
        action = "ALERT"

    return state, action


# ============================================================
# 4. LOAD ENVIRONMENTAL SENSOR DATA
# ============================================================

X = pd.read_csv("data/X_features.csv")

print("Sensor data loaded.")
print("Available sensor observations:", len(X))


# ============================================================
# 5. SIMULATE SEQUENTIAL PERCEPTION
# ============================================================

print("\n" + "=" * 70)
print("STARTING AGENT MONITORING")
print("=" * 70)


# Process first 10 sensor readings
for i in range(10):

    # --------------------------------------------------------
    # PERCEIVE
    # --------------------------------------------------------

    sensor_reading = X.iloc[[i]]

    print("\n" + "-" * 70)
    print(f"TIME STEP {i + 1}")
    print("-" * 70)

    print("\nPERCEIVE")
    print("Agent received new environmental sensor readings.")


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predicted_co = model.predict(sensor_reading)[0]

    print("\nPREDICT")
    print(f"Predicted CO concentration: {predicted_co:.2f}")


    # --------------------------------------------------------
    # DECIDE
    # --------------------------------------------------------

    state, action = make_decision(predicted_co)

    print("\nDECIDE")
    print(f"Pollution state : {state}")
    print(f"Agent action    : {action}")


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    print("\nACTION")

    if action == "MONITOR":

        print("Continue monitoring the environment.")

    elif action == "WARNING":

        print("Issue pollution WARNING.")

    else:

        print("Trigger pollution ALERT.")


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("AGENT MONITORING CYCLE COMPLETED")
print("=" * 70)