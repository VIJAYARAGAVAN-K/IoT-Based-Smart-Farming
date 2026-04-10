# ===============================
# ✅ Install dependencies in Colab
# ===============================
!pip install paho-mqtt tensorflow scikit-learn pandas numpy

import pandas as pd
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# ✅ Load dataset
# ===============================
df = pd.read_csv("/content/soil-moisture.csv")

if "Month" in df.columns:
    df["Month"] = df["Month"].astype("category").cat.codes

X = df.drop(columns=["avg_sm"])
y = df["avg_sm"]

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===============================
# ✅ Build LSTM model
# ===============================
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

print("🛠 Training LSTM model...")
model.fit(X_train, y_train, epochs=32, batch_size=16, verbose=1)

# ===============================
# ✅ MQTT Setup
# ===============================
broker = "broker.hivemq.com"   # free public broker
port = 1883
topic = "smartfarm/soil"

client = mqtt.Client()
client.connect(broker, port, 60)
client.loop_start()

# ===============================
# ✅ Thresholds (from averages you gave)
# ===============================
thresholds = {
    "avg_pm1": 1.36,
    "avg_pm2": 2.13,
    "avg_pm3": 49.11,
    "avg_am": 1.01,
    "avg_lum": 2722.15,
    "avg_temp": 22.53,
    "avg_humd": 73.01,
    "avg_pres": 93234.84,
    "avg_sm": 3257.61
}

print("🚀 Publishing predictions to MQTT topic:", topic)

# ===============================
# ✅ Infinite loop for publishing data
# ===============================
while True:
    # Simulate new sensor data
    new_data = {
        "Month": [0],
        "Day": [np.random.randint(1, 31)],
        "avg_pm1": [np.random.uniform(df["avg_pm1"].min(), df["avg_pm1"].max())],
        "avg_pm2": [np.random.uniform(df["avg_pm2"].min(), df["avg_pm2"].max())],
        "avg_pm3": [np.random.uniform(df["avg_pm3"].min(), df["avg_pm3"].max())],
        "avg_am": [np.random.uniform(0.5, 1.5)],
        "avg_lum": [np.random.uniform(df["avg_lum"].min(), df["avg_lum"].max())],
        "avg_temp": [np.random.uniform(df["avg_temp"].min(), df["avg_temp"].max())],
        "avg_humd": [np.random.uniform(df["avg_humd"].min(), df["avg_humd"].max())],
        "avg_pres": [np.random.uniform(df["avg_pres"].min(), df["avg_pres"].max())]
    }

    new_df = pd.DataFrame(new_data)[X.columns]
    new_scaled = scaler.transform(new_df)
    new_scaled = new_scaled.reshape((new_scaled.shape[0], 1, new_scaled.shape[1]))

    predicted_sm = model.predict(new_scaled, verbose=0)[0][0]

    # Decision rule
    decision = "Irrigation Needed" if predicted_sm < thresholds["avg_sm"] else "No Irrigation Required"

    # Create JSON message
    message = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "avg_pm1": round(float(new_data["avg_pm1"][0]), 2),
        "avg_pm2": round(float(new_data["avg_pm2"][0]), 2),
        "avg_pm3": round(float(new_data["avg_pm3"][0]), 2),
        "avg_am": round(float(new_data["avg_am"][0]), 2),
        "avg_lum": round(float(new_data["avg_lum"][0]), 2),
        "avg_temp": round(float(new_data["avg_temp"][0]), 2),
        "avg_humd": round(float(new_data["avg_humd"][0]), 2),
        "avg_pres": round(float(new_data["avg_pres"][0]), 2),
        "soil_moisture": round(float(predicted_sm), 2),
        "decision": decision
    }

    # Publish
    client.publish(topic, json.dumps(message))
    print("Published:", message)

    time.sleep(10)  # every 10s (use 3600 for 1 hr)
