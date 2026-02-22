#train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Paths
dataset_path = "../datasets/water_quality1.csv"  # replace with your CSV path
model_path = "../models/water_dl_model.keras"
scaler_path = "../models/water_scaler.pkl"

def map_iot_to_training_features(df):
    """
    Map your dataset columns to match IoT data field names
    This ensures consistency between training and real-world data
    """
    print("🔄 Mapping dataset columns to IoT data fields...")
    
    # Create a copy to avoid modifying original
    df_mapped = df.copy()
    
    # Check and rename columns to match IoT data structure
    column_mapping = {}
    
    # Map based on what exists in the dataset
    if 'pH' in df.columns and 'Ph' not in df.columns:
        column_mapping['pH'] = 'Ph'
        print("   📝 Renaming 'pH' to 'Ph'")
    elif 'Ph' in df.columns:
        print("   ✅ 'Ph' column already exists")
    
    if 'turbidity' in df.columns and 'Turbidity' not in df.columns:
        column_mapping['turbidity'] = 'Turbidity'
        print("   📝 Renaming 'turbidity' to 'Turbidity'")
    elif 'Turbidity' in df.columns:
        print("   ✅ 'Turbidity' column already exists")
    
    if 'temperature' in df.columns and 'Temperature' not in df.columns:
        column_mapping['temperature'] = 'Temperature'
        print("   📝 Renaming 'temperature' to 'Temperature'")
    elif 'Temperature' in df.columns:
        print("   ✅ 'Temperature' column already exists")
    
    if 'tds' in df.columns and 'Conductivity' not in df.columns:
        column_mapping['tds'] = 'Conductivity'
        print("   📝 Renaming 'tds' to 'Conductivity'")
    elif 'Conductivity' in df.columns:
        print("   ✅ 'Conductivity' column already exists")
    
    # Apply the renaming
    if column_mapping:
        df_mapped = df_mapped.rename(columns=column_mapping)
        print(f"   ✅ Applied {len(column_mapping)} column mappings")
    else:
        print("   ✅ No column mapping needed")
    
    return df_mapped

# Load dataset
print(f"📂 Loading dataset from {dataset_path}")
df = pd.read_csv(dataset_path)

print(f"📊 Original dataset columns: {df.columns.tolist()}")
print(f"📊 Original dataset shape: {df.shape}")

# Map IoT data fields to training features
df = map_iot_to_training_features(df)

print(f"📊 Mapped dataset columns: {df.columns.tolist()}")
print(f"📊 Mapped dataset shape: {df.shape}")

# Select only the features available from IoT
features = ['Ph', 'Turbidity', 'Temperature', 'Conductivity']

# Check if all required features exist
missing_features = [feat for feat in features if feat not in df.columns]
if missing_features:
    print(f"❌ Missing features in dataset: {missing_features}")
    print(f"✅ Available features: {df.columns.tolist()}")
    exit(1)

print(f"✅ Using features: {features}")

# Check target column
target = 'Label'  # replace with 'Potability' if your dataset uses that column
if target not in df.columns:
    print(f"❌ Target column '{target}' not found in dataset")
    print(f"✅ Available columns: {df.columns.tolist()}")
    # Try common target column names
    possible_targets = ['Potability', 'label', 'potability', 'Class', 'class']
    for possible_target in possible_targets:
        if possible_target in df.columns:
            target = possible_target
            print(f"🔄 Using alternative target: '{target}'")
            break
    else:
        exit(1)

print(f"✅ Using target: {target}")

# Prepare data
X = df[features].values
y = df[target].values

print(f"📊 Feature matrix shape: {X.shape}")
print(f"📊 Target vector shape: {y.shape}")
print(f"📊 Class distribution: {pd.Series(y).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for future predictions
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)
print("✅ Scaler saved successfully!")

# Build a simple neural network
model = Sequential([
    Dense(32, input_dim=len(features), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🧠 Model architecture:")
model.summary()

# Train model
print("🚀 Training model...")
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Evaluate
print("📊 Evaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.3f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"✅ Model saved: {model_path}")

# Print feature importance for IoT monitoring
print("\n🔍 Feature Mapping for IoT Data:")
print("   IoT Field      → Training Feature")
print("   -------------  → ----------------")
print("   'pH'           → 'Ph'")
print("   'turbidity'    → 'Turbidity'")
print("   'temperature'  → 'Temperature'")
print("   'tds'          → 'Conductivity'")

print("\n🎯 Training completed! Ready for IoT predictions.")