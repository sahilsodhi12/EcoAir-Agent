# 🌿 EcoAir Agent

### AI-Based Environmental Monitoring & Decision System

EcoAir Agent is an AI-based environmental monitoring system that uses historical air-quality sensor data to predict **CO concentration** and make an automated monitoring decision.

The project combines a **learned Random Forest regression model** with an **agent decision policy** and a **Streamlit dashboard**.

---

## 🎯 Project Goal

The goal of EcoAir Agent is to:

1. Perceive environmental sensor readings
2. Use a learned machine learning model to predict CO concentration
3. Interpret the predicted CO level
4. Select an appropriate monitoring action

### Agent Flow

```text
Environment
     ↓
PERCEIVE
     ↓
Sensor Readings
     ↓
RANDOM FOREST
     ↓
Predicted CO
     ↓
DECISION POLICY
     ↓
Monitor / Warning / Alert
     ↓
Environment
