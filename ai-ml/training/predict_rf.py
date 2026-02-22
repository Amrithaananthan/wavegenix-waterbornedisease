import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import json
import time
import sys

# Add blockchain messaging path
sys.path.append('D:\\newsih\\sihdeeo\\blockchain_messaging')
from message_sender import message_sender

# Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://wavegenix-6-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# ML Model paths
model_path = os.path.join("..", "models", "water_dl_model.keras")
scaler_path = os.path.join("..", "models", "water_scaler.pkl")

# Check if model files exist
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    print("💡 Please check if the model file exists")
    exit(1)

if not os.path.exists(scaler_path):
    print(f"❌ Scaler file not found: {scaler_path}")
    print("💡 Please check if the scaler file exists")
    exit(1)

# Load trained model and scaler
print(f"📂 Loading model: {model_path}")
model = load_model(model_path)
print(f"📂 Loading scaler: {scaler_path}")
scaler = joblib.load(scaler_path)

# Features in the same order as training
features = ['Ph', 'Turbidity', 'Temperature', 'Conductivity']

print("✅ AI Model loaded successfully!")
print("🎯 Ready for real-time predictions...")

def get_current_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def is_real_iot_data(data):
    """Check if this is real IoT sensor data (not prediction results)"""
    if not isinstance(data, dict):
        return False
    
    # Real IoT data has these specific fields
    required_fields = ['pH', 'tds', 'turbidity', 'temperature', 'deviceId']
    
    # Check if it's nested under waterData
    if 'waterData' in data and isinstance(data['waterData'], dict):
        water_data = data['waterData']
        return all(field in water_data for field in required_fields)
    
    # Check if it's direct IoT data
    return all(field in data for field in required_fields)

def extract_sensor_data(data):
    """Extract sensor values from IoT data"""
    try:
        if 'waterData' in data and isinstance(data['waterData'], dict):
            return data['waterData']
        else:
            return data
    except:
        return {}

def predict_potability(sensor_data):
    """Predict water potability from real sensor data"""
    try:
        # Map IoT data fields to training features
        feature_mapping = {
            'Ph': sensor_data.get('pH', 7.0),
            'Turbidity': sensor_data.get('turbidity', 0),
            'Temperature': sensor_data.get('temperature', 25.0),
            'Conductivity': sensor_data.get('tds', 0)  # Using TDS as Conductivity
        }
        
        print(f"   🔄 Feature mapping:")
        print(f"      pH: {sensor_data.get('pH')} → Ph: {feature_mapping['Ph']}")
        print(f"      turbidity: {sensor_data.get('turbidity')} → Turbidity: {feature_mapping['Turbidity']}")
        print(f"      temperature: {sensor_data.get('temperature')} → Temperature: {feature_mapping['Temperature']}")
        print(f"      tds: {sensor_data.get('tds')} → Conductivity: {feature_mapping['Conductivity']}")
        
        # Convert input to numpy array in correct order
        X = np.array([[float(feature_mapping[feat]) for feat in features]])
        
        # Scale using the same scaler used during training
        X_scaled = scaler.transform(X)
        
        # Predict using trained model
        pred = model.predict(X_scaled, verbose=0)[0][0]
        label = pred > 0.5  # True/False
        
        # Adjust confidence for better display
        confidence = max(pred, 1-pred)
        if confidence < 0.001:  # If confidence is extremely low
            confidence = 0.001  # Set minimum display value
        
        return label, float(confidence)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

def get_detailed_analysis(sensor_data, prediction, confidence):
    """Provide detailed water safety analysis with timestamps"""
    ph = sensor_data.get('pH', 7.0)
    tds = sensor_data.get('tds', 0)
    turbidity = sensor_data.get('turbidity', 0)
    temperature = sensor_data.get('temperature', 25.0)
    
    analysis = {
        'timestamp': get_current_timestamp(),
        'device_id': sensor_data.get('deviceId', 'Unknown'),
        'prediction': 'SAFE' if prediction else 'UNSAFE',
        'confidence': confidence,
        'parameters': {
            'pH': ph,
            'tds': tds,
            'turbidity': turbidity,
            'temperature': temperature
        },
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Detailed analysis
    if ph < 6.5:
        analysis['issues'].append(f"Acidic pH ({ph:.2f}) - should be 6.5-8.5")
    elif ph > 8.5:
        analysis['issues'].append(f"Alkaline pH ({ph:.2f}) - should be 6.5-8.5")
    else:
        analysis['recommendations'].append("pH level is optimal")
    
    if turbidity > 50:
        analysis['issues'].append(f"Very high turbidity ({turbidity:.2f} NTU) - should be <5 NTU")
    elif turbidity > 10:
        analysis['issues'].append(f"High turbidity ({turbidity:.2f} NTU) - should be <5 NTU")
    elif turbidity > 5:
        analysis['warnings'].append(f"Moderate turbidity ({turbidity:.2f} NTU) - borderline")
    else:
        analysis['recommendations'].append("Excellent water clarity")
    
    if tds > 1000:
        analysis['issues'].append(f"Very high TDS ({tds} ppm) - should be <500 ppm")
    elif tds > 500:
        analysis['warnings'].append(f"High TDS ({tds} ppm) - moderate concern")
    else:
        analysis['recommendations'].append("TDS within safe range")
    
    # Temperature context
    if temperature > 30:
        analysis['warnings'].append(f"Warm temperature ({temperature}°C) - may promote bacterial growth")
    elif temperature < 10:
        analysis['warnings'].append(f"Cold temperature ({temperature}°C) - may affect treatment")
    else:
        analysis['recommendations'].append("Optimal temperature range")
    
    return analysis

def trigger_emergency_alert(sensor_data, prediction, confidence, analysis):
    """Trigger emergency SMS when ML detects unsafe water"""
    if not prediction:  # If water is NOT POTABLE
        print(f"🚨 UNSAFE WATER DETECTED! Confidence: {confidence:.3f}")
        
        # Build issues list for SMS
        issues_text = "\n".join([f"• {issue}" for issue in analysis['issues']])
        
        emergency_msg = f"""🚨 EMERGENCY - WATER CRISIS 🚨

CONTAMINATION DETECTED:
• pH: {sensor_data.get('pH', 'N/A')}
• Turbidity: {sensor_data.get('turbidity', 'N/A')} NTU
• TDS: {sensor_data.get('tds', 'N/A')} ppm
• Temperature: {sensor_data.get('temperature', 'N/A')}°C
• Device: {sensor_data.get('deviceId', 'Unknown')}

CRITICAL ISSUES:
{issues_text}

🚫 DO NOT DRINK THIS WATER!
✅ USE BOTTLED WATER IMMEDIATELY
🏥 CONTACT HEALTH AUTHORITIES

AI Water Monitoring System
Confidence: {confidence:.3f}
Time: {analysis['timestamp']}"""

        print("📱 Triggering emergency SMS alerts...")
        try:
            success = message_sender.send_emergency_alert(emergency_msg)
            
            if success:
                print("✅ Emergency alerts dispatched successfully!")
            else:
                print("❌ Failed to send emergency alerts (Twilio daily limit reached)")
            
            return success
        except Exception as e:
            print(f"⚠  SMS error: {e}")
            return False
    else:
        print(f"✅ Water is SAFE. Confidence: {confidence:.3f}")
        return False

def process_with_ml(data):
    """Process data with ML model and trigger alerts"""
    try:
        # Extract features and make prediction
        prediction, confidence = predict_potability(data)
        
        # Get detailed analysis
        analysis = get_detailed_analysis(data, prediction, confidence)
        
        # Trigger emergency alerts if water is unsafe
        if prediction is not None:
            alert_sent = trigger_emergency_alert(data, prediction, confidence, analysis)
            if alert_sent:
                print("🎯 ALERT SYSTEM ACTIVATED - Check phones for SMS!")
        
        return prediction, confidence, analysis
        
    except Exception as e:
        print(f"❌ ML processing error: {e}")
        return None, None, None

def check_current_firebase_data():
    """Check what data is currently in Firebase"""
    print("\n🔍 Checking current Firebase data...")
    try:
        ref = db.reference('/')
        snapshot = ref.get()
        
        if not snapshot:
            print("   ❌ No data found in Firebase")
            return 0
        
        iot_count = 0
        other_count = 0
        
        for record_key, record_data in snapshot.items():
            if is_real_iot_data(record_data):
                iot_count += 1
                sensor_data = extract_sensor_data(record_data)
                print(f"   📱 IoT Device: {sensor_data.get('deviceId', 'Unknown')}")
                print(f"      pH: {sensor_data.get('pH')}, Turbidity: {sensor_data.get('turbidity')} NTU")
            else:
                other_count += 1
                print(f"   📄 Other data: {record_key}")
        
        print(f"\n📊 Firebase Summary:")
        print(f"   ✅ IoT Devices: {iot_count}")
        print(f"   📄 Other Records: {other_count}")
        print(f"   📈 Total Records: {len(snapshot)}")
        
        return iot_count
        
    except Exception as e:
        print(f"   ❌ Error checking Firebase: {e}")
        return 0

def process_real_iot_data():
    """Process only real IoT sensor data from Firebase"""
    try:
        ref = db.reference('/')
        snapshot = ref.get()
        
        if not snapshot:
            print("❌ No data found in Firebase")
            return
        
        print(f"📊 Found {len(snapshot)} records in database")
        
        real_iot_count = 0
        alert_triggered = False
        
        # Process each record
        for record_key, record_data in snapshot.items():
            # Skip ml_predictions and non-IoT data
            if record_key == 'ml_predictions' or not is_real_iot_data(record_data):
                continue
            
            real_iot_count += 1
            sensor_data = extract_sensor_data(record_data)
            
            print(f"\n🔍 Analyzing real IoT data: {record_key}")
            print(f"   📊 Sensor Readings:")
            print(f"      Device: {sensor_data.get('deviceId', 'Unknown')}")
            print(f"      pH: {sensor_data.get('pH')}")
            print(f"      TDS: {sensor_data.get('tds')} ppm")
            print(f"      Turbidity: {sensor_data.get('turbidity')} NTU")
            print(f"      Temperature: {sensor_data.get('temperature')}°C")
            
            # Process with ML and trigger alerts
            prediction, confidence, analysis = process_with_ml(sensor_data)
            
            if prediction is not None:
                result = "POTABLE" if prediction else "NOT POTABLE"
                print(f"   💧 ML Prediction: {result}")
                print(f"   🎯 Confidence: {confidence:.3f}")
                
                # Display detailed analysis
                print(f"   🔍 DETAILED ANALYSIS:")
                if analysis['issues']:
                    print(f"      🚨 CRITICAL ISSUES:")
                    for issue in analysis['issues']:
                        print(f"         • {issue}")
                
                if analysis['warnings']:
                    print(f"      ⚠  WARNINGS:")
                    for warning in analysis['warnings']:
                        print(f"         • {warning}")
                
                if analysis['recommendations']:
                    print(f"      ✅ RECOMMENDATIONS:")
                    for rec in analysis['recommendations']:
                        print(f"         • {rec}")
                
                print(f"   ⏰ Analysis Time: {analysis['timestamp']}")
                
                if not prediction:
                    alert_triggered = True
        
        if real_iot_count == 0:
            print("\n❌ No real IoT sensor data found!")
            print("   Looking for data with: deviceId, pH, tds, turbidity, temperature")
        else:
            print(f"\n✅ Processed {real_iot_count} real IoT data records")
            if alert_triggered:
                print("🚨 EMERGENCY ALERTS WERE SENT FOR UNSAFE WATER!")
        
        return alert_triggered

    except Exception as e:
        print(f"❌ Error processing IoT data: {e}")
        return False

def real_time_iot_listener(event):
    """Real-time listener - ONLY for new IoT sensor data"""
    # Skip if data is None or path contains ml_predictions
    if event.data is None:
        return
        
    if event.path and '/ml_predictions' in event.path:
        return
    
    if not is_real_iot_data(event.data):
        return
    
    sensor_data = extract_sensor_data(event.data)
    current_time = get_current_timestamp()
    
    print(f"\n" + "="*70)
    print(f"🔄 NEW REAL-TIME IoT DATA RECEIVED:")
    print(f"   📍 Device: {sensor_data.get('deviceId', 'Unknown')}")
    print(f"   ⏰ Detection Time: {current_time}")
    print(f"   📊 Sensor Readings:")
    print(f"      pH: {sensor_data.get('pH')}")
    print(f"      TDS: {sensor_data.get('tds')} ppm") 
    print(f"      Turbidity: {sensor_data.get('turbidity')} NTU")
    print(f"      Temperature: {sensor_data.get('temperature')}°C")
    
    # Process with ML and trigger real-time alerts
    prediction, confidence, analysis = process_with_ml(sensor_data)
    
    if prediction is not None:
        result = "POTABLE" if prediction else "NOT POTABLE"
        print(f"   🤖 REAL-TIME PREDICTION: {result}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        
        # Display detailed analysis
        print(f"   🔍 DETAILED ANALYSIS:")
        if analysis['issues']:
            print(f"      🚨 CRITICAL ISSUES:")
            for issue in analysis['issues']:
                print(f"         • {issue}")
        
        if analysis['warnings']:
            print(f"      ⚠  WARNINGS:")
            for warning in analysis['warnings']:
                print(f"         • {warning}")
        
        if analysis['recommendations']:
            print(f"      ✅ RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"         • {rec}")
        
        if not prediction:
            print("   🚨 REAL-TIME EMERGENCY ALERT ACTIVATED!")
    
    print("="*70)

# Main execution - CORRECTED LINE BELOW
if __name__ == "__main__":
    print("🚀 Starting Water Potability Prediction System")
    print("🎯 Integrated with Emergency SMS Alerts")
    print("📡 Connecting to Firebase...")
    
    # Check current Firebase data first
    iot_count = check_current_firebase_data()
    
    # Process existing real IoT data
    print("\n📊 Scanning for real IoT sensor data...")
    alert_sent = process_real_iot_data()
    
    # Start real-time monitoring
    print("\n🎧 Starting real-time monitoring...")
    print("📍 Listening for NEW IoT sensor data...")
    print("⏰ System time:", get_current_timestamp())
    
    if iot_count == 0:
        print("\n💡 TROUBLESHOOTING:")
        print("   • Ensure your IoT device is sending data to Firebase")
        print("   • Data should include: deviceId, pH, tds, turbidity, temperature")
        print("   • The system will detect NEW data automatically")
    
    # Listen for new data
    try:
        ref = db.reference('/')
        ref.listen(real_time_iot_listener)
        
        print("\n✅ System running! Waiting for NEW IoT sensor data...")
        print("   Emergency SMS alerts are ACTIVE and READY!")
        print("   Press Ctrl+C to stop")
        
        # Keep the program running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping real-time monitoring...")
        print("👋 System shutdown complete")
    except Exception as e:
        print(f"❌ Error in real-time monitoring: {e}")