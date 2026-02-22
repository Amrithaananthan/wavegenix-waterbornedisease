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
model_path = os.path.join("..", "models", "intelligent_water_ai.h5")
scaler_path = os.path.join("..", "models", "ai_scaler.pkl")
feature_importance_path = os.path.join("..", "models", "feature_importance.json")

# Check if model files exist
if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    print("💡 Please run train_model.py first to create the intelligent AI model")
    exit(1)

if not os.path.exists(scaler_path):
    print(f"❌ Scaler file not found: {scaler_path}")
    print("💡 Please run train_model.py first to create the AI scaler")
    exit(1)

# Load trained model and scaler
print("📂 Loading Intelligent Water AI System...")
model = load_model(model_path)
print("📂 Loading AI Scaler...")
scaler = joblib.load(scaler_path)

# Load feature importance
try:
    with open(feature_importance_path, 'r') as f:
        feature_importance = json.load(f)
    print("📂 Loaded feature importance analysis")
except:
    feature_importance = None
    print("⚠️  Feature importance file not found")

# Get the expected number of features from the scaler
expected_features = scaler.n_features_in_
print(f"🔢 Scaler expects {expected_features} features")

# Features that your model was trained with
features = ['Ph', 'Turbidity', 'Temperature', 'Conductivity', 'Dissolved_Oxygen', 'TDS', 'pH_Variation', 'Quality_Score']

print("✅ AI Model loaded successfully!")
print("🎯 Ready for real-time predictions...")
print(f"📊 Monitoring {len(features)} water quality parameters")

# Bacterial and Microorganism Prediction Database
BACTERIA_DATABASE = {
    'e_coli': {
        'name': 'E. coli',
        'do_threshold': 5.0,
        'temp_range': (20, 45),
        'ph_range': (4.5, 9.0),
        'turbidity_threshold': 10,
        'diseases': ['Gastroenteritis', 'UTI', 'Meningitis'],
        'risk_level': 'HIGH'
    },
    'coliform': {
        'name': 'Coliform Bacteria',
        'do_threshold': 4.0,
        'temp_range': (10, 45),
        'ph_range': (5.0, 9.0),
        'turbidity_threshold': 5,
        'diseases': ['Diarrhea', 'Dysentery', 'Typhoid'],
        'risk_level': 'HIGH'
    },
    'legionella': {
        'name': 'Legionella',
        'do_threshold': 2.0,
        'temp_range': (25, 42),
        'ph_range': (5.5, 8.5),
        'turbidity_threshold': 15,
        'diseases': ['Legionnaires Disease', 'Pontiac Fever'],
        'risk_level': 'VERY HIGH'
    },
    'vibrio_cholerae': {
        'name': 'Vibrio Cholerae',
        'do_threshold': 3.0,
        'temp_range': (20, 40),
        'ph_range': (6.0, 9.0),
        'turbidity_threshold': 20,
        'diseases': ['Cholera'],
        'risk_level': 'VERY HIGH'
    },
    'salmonella': {
        'name': 'Salmonella',
        'do_threshold': 4.5,
        'temp_range': (15, 45),
        'ph_range': (4.5, 9.0),
        'turbidity_threshold': 12,
        'diseases': ['Salmonellosis', 'Typhoid Fever'],
        'risk_level': 'HIGH'
    },
    'pseudomonas': {
        'name': 'Pseudomonas',
        'do_threshold': 6.0,
        'temp_range': (20, 42),
        'ph_range': (5.5, 8.5),
        'turbidity_threshold': 8,
        'diseases': ['Skin Infections', 'Respiratory Issues'],
        'risk_level': 'MEDIUM'
    },
    'algae_bloom': {
        'name': 'Algae Bloom',
        'do_threshold': 12.0,  # High DO indicates algal activity
        'temp_range': (15, 30),
        'ph_range': (7.5, 9.0),
        'turbidity_threshold': 25,
        'diseases': ['Skin Rashes', 'Gastrointestinal Issues', 'Liver Damage'],
        'risk_level': 'HIGH'
    }
}

def get_current_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def is_real_iot_data(data):
    """Check if this is real IoT sensor data (not prediction results)"""
    if not isinstance(data, dict):
        return False
    
    # Check for your actual sensor data structure - looking for lowercase fields
    sensor_fields = ['pH', 'temperature', 'tds', 'turbidity', 'dissolvedOxygen']
    return any(field in data for field in sensor_fields)

def extract_sensor_data(data):
    """Extract sensor values from IoT data"""
    try:
        # Your data is directly in the root, no nested waterData
        return data
    except:
        return {}

def predict_potability(sensor_data):
    """Predict water potability from real sensor data"""
    try:
        # Map your ACTUAL sensor data fields to model features
        ph_value = float(sensor_data.get('pH', 7.0))
        turbidity_value = float(sensor_data.get('turbidity', 0.0))
        temperature_value = float(sensor_data.get('temperature', 25.0))
        tds_value = float(sensor_data.get('tds', 0.0))
        dissolved_oxygen = float(sensor_data.get('dissolvedOxygen', 8.0))
        quality_score = float(sensor_data.get('qualityScore', 100.0))
        
        # Calculate additional features to match the 8 expected by scaler
        ph_variation = abs(ph_value - 7.0)  # Variation from neutral pH
        
        # Debug: Print what we're receiving
        print(f"   🔍 Raw sensor data received:")
        print(f"      pH: {ph_value}")
        print(f"      turbidity: {turbidity_value}")
        print(f"      temperature: {temperature_value}")
        print(f"      tds: {tds_value}")
        print(f"      dissolvedOxygen: {dissolved_oxygen}")
        print(f"      qualityScore: {quality_score}")
        print(f"      pH Variation: {ph_variation}")
        print(f"      sensorInWater: {sensor_data.get('sensorInWater', 'N/A')}")
        
        # Create feature mapping for ALL 8 expected features
        feature_mapping = {
            'Ph': ph_value,
            'Turbidity': turbidity_value,
            'Temperature': temperature_value,
            'Conductivity': tds_value,  # Using TDS as Conductivity
            'Dissolved_Oxygen': dissolved_oxygen,
            'TDS': tds_value,  # Same as Conductivity for now
            'pH_Variation': ph_variation,
            'Quality_Score': quality_score
        }
        
        print(f"   🔄 Feature mapping for model (8 features):")
        for feat, val in feature_mapping.items():
            print(f"      {feat}: {val}")
        
        # Convert input to numpy array in correct order (8 features)
        X = np.array([[float(feature_mapping[feat]) for feat in features]])
        
        print(f"   📊 Input shape for scaler: {X.shape}")
        
        # Scale using the same scaler used during training
        X_scaled = scaler.transform(X)
        
        # Predict using trained model
        pred = model.predict(X_scaled, verbose=0)[0][0]
        label = pred > 0.5  # True/False
        
        # Adjust confidence for better display
        confidence = max(pred, 1-pred)
        if confidence < 0.001:
            confidence = 0.001
        
        return label, float(confidence)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_bacteria_from_do(sensor_data):
    """Predict bacterial and microorganism presence based on dissolved oxygen and other parameters"""
    try:
        ph = float(sensor_data.get('pH', 7.0))
        turbidity = float(sensor_data.get('turbidity', 0.0))
        temperature = float(sensor_data.get('temperature', 25.0))
        dissolved_oxygen = float(sensor_data.get('dissolvedOxygen', 8.0))
        
        detected_bacteria = []
        
        for bacteria_id, bacteria_info in BACTERIA_DATABASE.items():
            # Check dissolved oxygen threshold
            do_condition = dissolved_oxygen <= bacteria_info['do_threshold']
            
            # Check for algal bloom (high DO)
            if bacteria_id == 'algae_bloom':
                do_condition = dissolved_oxygen >= bacteria_info['do_threshold']
            
            # Check temperature range
            temp_min, temp_max = bacteria_info['temp_range']
            temp_condition = temp_min <= temperature <= temp_max
            
            # Check pH range
            ph_min, ph_max = bacteria_info['ph_range']
            ph_condition = ph_min <= ph <= ph_max
            
            # Check turbidity
            turbidity_condition = turbidity >= bacteria_info['turbidity_threshold']
            
            # Calculate probability based on conditions
            conditions_met = sum([do_condition, temp_condition, ph_condition, turbidity_condition])
            probability = conditions_met / 4.0
            
            if probability >= 0.5:  # At least 2 conditions met
                detected_bacteria.append({
                    'id': bacteria_id,
                    'name': bacteria_info['name'],
                    'probability': probability,
                    'diseases': bacteria_info['diseases'],
                    'risk_level': bacteria_info['risk_level'],
                    'conditions_met': conditions_met,
                    'do_level': dissolved_oxygen,
                    'do_threshold': bacteria_info['do_threshold']
                })
        
        # Sort by probability (highest first)
        detected_bacteria.sort(key=lambda x: x['probability'], reverse=True)
        return detected_bacteria
        
    except Exception as e:
        print(f"❌ Bacteria prediction error: {e}")
        return []

def get_detailed_analysis(sensor_data, prediction, confidence):
    """Provide detailed water safety analysis with disease detection"""
    ph = float(sensor_data.get('pH', 7.0))
    tds = float(sensor_data.get('tds', 0))
    turbidity = float(sensor_data.get('turbidity', 0))
    temperature = float(sensor_data.get('temperature', 25.0))
    
    # Enhanced parameters from your actual data
    dissolved_oxygen = sensor_data.get('dissolvedOxygen', 8.0)
    quality_score = sensor_data.get('qualityScore', 100.0)
    status = sensor_data.get('status', 'Unknown')
    sensor_in_water = sensor_data.get('sensorInWater', False)
    risk_level = sensor_data.get('riskLevel', 'Unknown')
    
    analysis = {
        'timestamp': get_current_timestamp(),
        'device_id': sensor_data.get('deviceId', 'Unknown'),
        'prediction': 'SAFE' if prediction else 'UNSAFE',
        'confidence': confidence,
        'parameters': {
            'pH': ph,
            'tds': tds,
            'turbidity': turbidity,
            'temperature': temperature,
            'dissolved_oxygen': dissolved_oxygen,
            'quality_score': quality_score,
            'status': status,
            'sensor_in_water': sensor_in_water,
            'risk_level': risk_level
        },
        'issues': [],
        'warnings': [],
        'recommendations': [],
        'potential_diseases': [],
        'detected_bacteria': []
    }
    
    # Detailed water quality analysis
    if ph < 6.5:
        analysis['issues'].append(f"Acidic pH ({ph:.2f}) - should be 6.5-8.5")
    elif ph > 8.5:
        analysis['issues'].append(f"Alkaline pH ({ph:.2f}) - should be 6.5-8.5")
    else:
        analysis['recommendations'].append("pH level is optimal")
    
    # Turbidity analysis
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
    
    # Enhanced parameter analysis with DO focus
    if dissolved_oxygen < 4:
        analysis['issues'].append(f"Very low dissolved oxygen ({dissolved_oxygen:.1f} mg/L) - HIGH bacterial risk")
    elif dissolved_oxygen < 5:
        analysis['issues'].append(f"Low dissolved oxygen ({dissolved_oxygen:.1f} mg/L) - bacterial growth likely")
    elif dissolved_oxygen > 12:
        analysis['warnings'].append(f"Supersaturated oxygen ({dissolved_oxygen:.1f} mg/L) - possible algal bloom")
    elif dissolved_oxygen > 14:
        analysis['issues'].append(f"Extremely high oxygen ({dissolved_oxygen:.1f} mg/L) - algal bloom confirmed")
    else:
        analysis['recommendations'].append("Good oxygen levels")
    
    # Sensor status warnings
    if not sensor_in_water:
        analysis['warnings'].append("Sensor not in water - readings may be inaccurate")
    
    # Temperature context
    if temperature > 30:
        analysis['warnings'].append(f"Warm temperature ({temperature}°C) - promotes bacterial growth")
    elif temperature < 10:
        analysis['warnings'].append(f"Cold temperature ({temperature}°C) - may affect treatment")
    else:
        analysis['recommendations'].append("Optimal temperature range")
    
    # Quality score analysis
    if quality_score < 60:
        analysis['issues'].append(f"Poor quality score ({quality_score:.1f}/100)")
    elif quality_score < 80:
        analysis['warnings'].append(f"Moderate quality score ({quality_score:.1f}/100)")
    else:
        analysis['recommendations'].append(f"Excellent quality score ({quality_score:.1f}/100)")
    
    # Risk level analysis
    if risk_level and risk_level.lower() != 'low':
        analysis['warnings'].append(f"Elevated risk level: {risk_level}")
    
    # Bacteria and disease detection
    analysis['detected_bacteria'] = predict_bacteria_from_do(sensor_data)
    analysis['potential_diseases'] = detect_waterborne_diseases(analysis['parameters'], analysis['detected_bacteria'])
    
    return analysis

def detect_waterborne_diseases(parameters, detected_bacteria):
    """Detect potential waterborne diseases based on water quality parameters and bacteria detection"""
    ph = parameters.get('pH', 7.0)
    turbidity = parameters.get('turbidity', 0)
    tds = parameters.get('tds', 0)
    temperature = parameters.get('temperature', 25.0)
    dissolved_oxygen = parameters.get('dissolved_oxygen', 8.0)
    
    diseases = []
    
    # Disease detection based on bacteria
    for bacteria in detected_bacteria:
        for disease_name in bacteria['diseases']:
            diseases.append({
                'name': disease_name,
                'risk': bacteria['risk_level'],
                'causes': [f"Presence of {bacteria['name']} bacteria"],
                'symptoms': get_disease_symptoms(disease_name),
                'prevention': get_disease_prevention(disease_name),
                'source_bacteria': bacteria['name'],
                'probability': bacteria['probability']
            })
    
    # Additional disease detection based on water parameters
    if turbidity > 5 and dissolved_oxygen < 5:
        diseases.append({
            'name': 'General Gastrointestinal Infections',
            'risk': 'HIGH',
            'causes': ['High turbidity with low oxygen indicates bacterial contamination'],
            'symptoms': ['Diarrhea', 'Stomach cramps', 'Nausea', 'Vomiting'],
            'prevention': ['Boil water before drinking', 'Use water filters', 'Avoid direct consumption'],
            'source_bacteria': 'Unknown (General Bacterial)',
            'probability': 0.7
        })
    
    if dissolved_oxygen < 3:
        diseases.append({
            'name': 'Anaerobic Bacterial Infections',
            'risk': 'HIGH',
            'causes': ['Extremely low oxygen favors anaerobic bacteria growth'],
            'symptoms': ['Severe gastrointestinal distress', 'Fever', 'Dehydration'],
            'prevention': ['Emergency water purification', 'Medical attention required'],
            'source_bacteria': 'Anaerobic Bacteria',
            'probability': 0.8
        })
    
    if dissolved_oxygen > 12:
        diseases.append({
            'name': 'Algal Toxin Exposure',
            'risk': 'MEDIUM',
            'causes': ['High oxygen levels indicate possible algal bloom'],
            'symptoms': ['Skin rashes', 'Liver problems', 'Neurological issues'],
            'prevention': ['Avoid contact with water', 'Use alternative water sources'],
            'source_bacteria': 'Algae/Cyanobacteria',
            'probability': 0.6
        })
    
    # Remove duplicates
    unique_diseases = []
    seen_diseases = set()
    for disease in diseases:
        if disease['name'] not in seen_diseases:
            unique_diseases.append(disease)
            seen_diseases.add(disease['name'])
    
    return unique_diseases

def get_disease_symptoms(disease_name):
    """Get symptoms for specific diseases"""
    symptom_map = {
        'Gastroenteritis': ['Diarrhea', 'Vomiting', 'Stomach cramps', 'Fever'],
        'UTI': ['Burning sensation', 'Frequent urination', 'Pelvic pain'],
        'Meningitis': ['Headache', 'Fever', 'Stiff neck', 'Nausea'],
        'Diarrhea': ['Loose stools', 'Dehydration', 'Stomach cramps'],
        'Dysentery': ['Bloody diarrhea', 'Fever', 'Stomach pain'],
        'Typhoid': ['High fever', 'Headache', 'Stomach pain', 'Loss of appetite'],
        'Legionnaires Disease': ['Cough', 'Fever', 'Shortness of breath', 'Muscle aches'],
        'Pontiac Fever': ['Fever', 'Muscle aches', 'Headache'],
        'Cholera': ['Severe diarrhea', 'Dehydration', 'Vomiting', 'Muscle cramps'],
        'Salmonellosis': ['Diarrhea', 'Fever', 'Stomach cramps'],
        'Typhoid Fever': ['Sustained fever', 'Weakness', 'Stomach pain', 'Headache'],
        'Skin Infections': ['Rashes', 'Itching', 'Redness', 'Swelling'],
        'Respiratory Issues': ['Coughing', 'Wheezing', 'Shortness of breath'],
        'Skin Rashes': ['Itching', 'Redness', 'Blisters'],
        'Gastrointestinal Issues': ['Nausea', 'Vomiting', 'Diarrhea'],
        'Liver Damage': ['Jaundice', 'Abdominal pain', 'Fatigue']
    }
    return symptom_map.get(disease_name, ['Fever', 'Fatigue', 'General discomfort'])

def get_disease_prevention(disease_name):
    """Get prevention methods for specific diseases"""
    prevention_map = {
        'Gastroenteritis': ['Boil water', 'Practice good hygiene', 'Use water filters'],
        'Cholera': ['Emergency water treatment', 'Medical attention', 'Hydration'],
        'Typhoid': ['Vaccination', 'Water purification', 'Proper sanitation'],
        'Legionnaires Disease': ['Water system maintenance', 'Avoid aerosol exposure'],
        'Skin Infections': ['Avoid contaminated water', 'Use protective clothing']
    }
    return prevention_map.get(disease_name, ['Boil water before use', 'Seek medical advice', 'Use alternative water sources'])

def trigger_emergency_alert(sensor_data, prediction, confidence, analysis):
    """Trigger emergency SMS when ML detects unsafe water"""
    if not prediction:  # If water is NOT POTABLE
        print(f"🚨 UNSAFE WATER DETECTED! Confidence: {confidence:.3f}")
        
        # Build issues and diseases list for SMS
        issues_text = "\n".join([f"• {issue}" for issue in analysis['issues']])
        if not issues_text:
            issues_text = "• AI model detected contamination"
        
        # Add bacteria detection
        bacteria_text = ""
        if analysis['detected_bacteria']:
            bacteria_text = "\n🦠 DETECTED BACTERIA:\n"
            for bacteria in analysis['detected_bacteria'][:3]:
                bacteria_text += f"• {bacteria['name']} ({bacteria['risk_level']} Risk)\n"

        # Add disease warnings
        diseases_text = ""
        if analysis['potential_diseases']:
            diseases_text = "\n🚑 POTENTIAL DISEASES:\n"
            for disease in analysis['potential_diseases'][:3]:
                diseases_text += f"• {disease['name']} ({disease['risk']} Risk)\n"

        emergency_msg = f"""🚨 EMERGENCY - WATER CRISIS 🚨

CONTAMINATION DETECTED:
• pH: {sensor_data.get('pH', 'N/A')}
• Turbidity: {sensor_data.get('turbidity', 'N/A')} NTU
• TDS: {sensor_data.get('tds', 'N/A')} ppm
• Temperature: {sensor_data.get('temperature', 'N/A')}°C
• Dissolved Oxygen: {sensor_data.get('dissolvedOxygen', 'N/A')} mg/L
• Quality Score: {sensor_data.get('qualityScore', 'N/A')}/100

CRITICAL ISSUES:
{issues_text}
{bacteria_text}
{diseases_text}
🚫 DO NOT DRINK THIS WATER!
✅ USE BOTTLED WATER IMMEDIATELY
🏥 CONTACT HEALTH AUTHORITIES

🤖 Intelligent Water Monitoring System
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
            print(f"⚠️  SMS error: {e}")
            return False
    else:
        print(f"✅ Water is SAFE. Confidence: {confidence:.3f}")
        return False

def display_bacteria_analysis(detected_bacteria):
    """Display bacteria analysis in a formatted way"""
    if not detected_bacteria:
        print(f"      ✅ No significant bacterial contamination detected")
        return
    
    print(f"      🦠 BACTERIAL CONTAMINATION DETECTED:")
    for bacteria in detected_bacteria[:3]:  # Show top 3
        risk_color = "🟢" if bacteria['risk_level'] == 'LOW' else "🟡" if bacteria['risk_level'] == 'MEDIUM' else "🟠" if bacteria['risk_level'] == 'HIGH' else "🔴"
        print(f"         {risk_color} {bacteria['name']} ({bacteria['risk_level']} RISK)")
        print(f"            📊 Probability: {bacteria['probability']:.1%}")
        print(f"            💨 DO Level: {bacteria['do_level']:.1f} mg/L (Threshold: {bacteria['do_threshold']} mg/L)")
        print(f"            🎯 Conditions Met: {bacteria['conditions_met']}/4")

def display_disease_analysis(diseases):
    """Display disease analysis in a formatted way"""
    if not diseases:
        print(f"      ✅ No significant disease risks detected")
        return
    
    print(f"      🚑 DISEASE RISK ASSESSMENT:")
    for disease in diseases[:3]:  # Show top 3
        risk_color = "🟢" if disease['risk'] == 'LOW' else "🟡" if disease['risk'] == 'MEDIUM' else "🟠" if disease['risk'] == 'HIGH' else "🔴"
        print(f"         {risk_color} {disease['name']} ({disease['risk']} RISK)")
        print(f"            📍 Source: {disease['source_bacteria']}")
        print(f"            📊 Probability: {disease['probability']:.1%}")
        print(f"            🤒 Symptoms: {', '.join(disease['symptoms'][:3])}...")
        print(f"            🛡️  Prevention: {', '.join(disease['prevention'][:2])}")

# ... (rest of the functions remain the same as your original code, including process_with_ml, check_current_firebase_data, display_feature_importance, process_real_iot_data, and real_time_iot_listener)

def process_with_ml(data):
    """Process data with ML model and trigger alerts"""
    try:
        # Extract features and make prediction
        prediction, confidence = predict_potability(data)
        
        # Get detailed analysis with bacteria and disease detection
        analysis = get_detailed_analysis(data, prediction, confidence)
        
        # Trigger emergency alerts if water is unsafe
        if prediction is not None:
            alert_sent = trigger_emergency_alert(data, prediction, confidence, analysis)
            if alert_sent:
                print("🎯 ALERT SYSTEM ACTIVATED - Check phones for SMS!")
        
        return prediction, confidence, analysis
        
    except Exception as e:
        print(f"❌ ML processing error: {e}")
        import traceback
        traceback.print_exc()
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
                print(f"   📱 IoT Device Data: {record_key}")
                print(f"      🌡  Temperature: {sensor_data.get('temperature', 'N/A')}°C")
                print(f"      🧪 pH: {sensor_data.get('pH', 'N/A')}")
                print(f"      💧 TDS: {sensor_data.get('tds', 'N/A')} ppm")
                print(f"      💨 Dissolved Oxygen: {sensor_data.get('dissolvedOxygen', 'N/A')} mg/L")
                print(f"      🌫  Turbidity: {sensor_data.get('turbidity', 'N/A')} NTU")
                print(f"      💧 Quality Score: {sensor_data.get('qualityScore', 'N/A')}/100")
                print(f"      🚨 Status: {sensor_data.get('status', 'N/A')}")
                print(f"      💦 Sensor in Water: {sensor_data.get('sensorInWater', 'N/A')}")
                print(f"      ⚠️  Risk Level: {sensor_data.get('riskLevel', 'N/A')}")
            else:
                other_count += 1
                if record_key != 'ml_predictions':
                    print(f"   📄 Other data: {record_key}")
        
        print(f"\n📊 Firebase Summary:")
        print(f"   ✅ IoT Data Records: {iot_count}")
        print(f"   📄 Other Records: {other_count}")
        print(f"   📈 Total Records: {len(snapshot)}")
        
        return iot_count
        
    except Exception as e:
        print(f"   ❌ Error checking Firebase: {e}")
        return 0

def display_feature_importance():
    """Display feature importance analysis"""
    if not feature_importance:
        print(f"   🔍 Feature importance: Using {len(features)}-parameter model")
        return
    
    print(f"   🔍 FEATURE IMPORTANCE ANALYSIS:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features:
        stars = "★" * int(importance * 10)
        print(f"      • {feature}: {importance:.3f} {stars}")

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
            
            print(f"\n🔍 Analyzing IoT data: {record_key}")
            print(f"   📊 Sensor Readings:")
            print(f"      🌡  Temperature: {sensor_data.get('temperature', 'N/A')}°C")
            print(f"      🧪 pH: {sensor_data.get('pH', 'N/A')}")
            print(f"      💧 TDS: {sensor_data.get('tds', 'N/A')} ppm")
            print(f"      💨 Dissolved Oxygen: {sensor_data.get('dissolvedOxygen', 'N/A')} mg/L")
            print(f"      🌫  Turbidity: {sensor_data.get('turbidity', 'N/A')} NTU")
            print(f"      💧 Quality Score: {sensor_data.get('qualityScore', 'N/A')}/100")
            print(f"      🚨 Status: {sensor_data.get('status', 'N/A')}")
            print(f"      💦 Sensor in Water: {sensor_data.get('sensorInWater', 'N/A')}")
            print(f"      ⚠️  Risk Level: {sensor_data.get('riskLevel', 'N/A')}")
            
            # Process with ML and trigger alerts
            prediction, confidence, analysis = process_with_ml(sensor_data)
            
            if prediction is not None:
                result = "POTABLE" if prediction else "NOT POTABLE"
                print(f"   🤖 REAL-TIME PREDICTION: {result}")
                print(f"   🎯 Confidence: {confidence:.3f}")
                
                # Display detailed analysis
                print(f"   🔍 WATER QUALITY ANALYSIS:")
                if analysis['issues']:
                    print(f"      🚨 CRITICAL ISSUES:")
                    for issue in analysis['issues']:
                        print(f"         • {issue}")
                
                if analysis['warnings']:
                    print(f"      ⚠️  WARNINGS:")
                    for warning in analysis['warnings']:
                        print(f"         • {warning}")
                
                if analysis['recommendations']:
                    print(f"      ✅ RECOMMENDATIONS:")
                    for rec in analysis['recommendations']:
                        print(f"         • {rec}")
                
                # Display bacteria analysis
                display_bacteria_analysis(analysis['detected_bacteria'])
                
                # Display disease analysis
                display_disease_analysis(analysis['potential_diseases'])
                
                # Display feature importance
                display_feature_importance()
                
                print(f"   ⏰ Analysis Time: {analysis['timestamp']}")
                
                if not prediction:
                    alert_triggered = True
        
        if real_iot_count == 0:
            print("\n❌ No real IoT sensor data found!")
            print("   Looking for data with fields like: temperature, pH, tds, turbidity, etc.")
        else:
            print(f"\n✅ Processed {real_iot_count} IoT data records")
            if alert_triggered:
                print("🚨 EMERGENCY ALERTS WERE SENT FOR UNSAFE WATER!")
        
        return alert_triggered

    except Exception as e:
        print(f"❌ Error processing IoT data: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"   ⏰ Detection Time: {current_time}")
    print(f"   📊 Sensor Readings:")
    print(f"      🌡  Temperature: {sensor_data.get('temperature', 'N/A')}°C")
    print(f"      🧪 pH: {sensor_data.get('pH', 'N/A')}")
    print(f"      💧 TDS: {sensor_data.get('tds', 'N/A')} ppm")
    print(f"      💨 Dissolved Oxygen: {sensor_data.get('dissolvedOxygen', 'N/A')} mg/L")
    print(f"      🌫  Turbidity: {sensor_data.get('turbidity', 'N/A')} NTU")
    print(f"      💧 Quality Score: {sensor_data.get('qualityScore', 'N/A')}/100")
    print(f"      🚨 Status: {sensor_data.get('status', 'N/A')}")
    print(f"      💦 Sensor in Water: {sensor_data.get('sensorInWater', 'N/A')}")
    print(f"      ⚠️  Risk Level: {sensor_data.get('riskLevel', 'N/A')}")
    
    # Process with ML and trigger real-time alerts
    prediction, confidence, analysis = process_with_ml(sensor_data)
    
    if prediction is not None:
        result = "POTABLE" if prediction else "NOT POTABLE"
        print(f"   🤖 REAL-TIME PREDICTION: {result}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        
        # Display detailed analysis
        print(f"   🔍 WATER QUALITY ANALYSIS:")
        if analysis['issues']:
            print(f"      🚨 CRITICAL ISSUES:")
            for issue in analysis['issues']:
                print(f"         • {issue}")
        
        if analysis['warnings']:
            print(f"      ⚠️  WARNINGS:")
            for warning in analysis['warnings']:
                print(f"         • {warning}")
        
        # Display bacteria analysis
        display_bacteria_analysis(analysis['detected_bacteria'])
        
        # Display disease analysis
        display_disease_analysis(analysis['potential_diseases'])
        
        if not prediction:
            print("   🚨 REAL-TIME EMERGENCY ALERT ACTIVATED!")
    
    print("="*70)

# Main execution
if __name__ == "__main__":
    print("🚀 INTELLIGENT WATER QUALITY MONITORING SYSTEM")
    print("🎯 Enhanced Neural Network AI + Bacteria Detection")
    print("🦠 Advanced Waterborne Disease Risk Assessment")
    print("💨 Dissolved Oxygen-Based Bacterial Prediction")
    print("📡 Real-time Firebase Integration")
    print(f"🔢 Using {len(features)}-feature model")
    print("=" * 70)
    
    # Check current Firebase data first
    iot_count = check_current_firebase_data()
    
    # Process existing real IoT data
    print("\n📊 Scanning for IoT sensor data...")
    alert_sent = process_real_iot_data()
    
    # Start real-time monitoring
    print("\n🎧 Starting real-time monitoring...")
    print("📍 Listening for NEW IoT sensor data...")
    print("⏰ System time:", get_current_timestamp())
    
    if iot_count == 0:
        print("\n💡 TROUBLESHOOTING:")
        print("   • Ensure your IoT device is sending data to Firebase")
        print("   • Data should include: temperature, pH, tds, turbidity, dissolvedOxygen")
        print("   • Enhanced parameters: qualityScore, sensorInWater")
        print("   • The system will detect NEW data automatically")
    
    # Listen for new data
    try:
        ref = db.reference('/')
        ref.listen(real_time_iot_listener)
        
        print("\n✅ INTELLIGENT SYSTEM RUNNING!")
        print(f"   🤖 {len(features)}-Parameter Neural Network AI")
        print("   🦠 Bacterial Contamination Detection")
        print("   💨 DO-Based Disease Prediction")
        print("   🔍 Feature Importance Analysis") 
        print("   📱 Emergency SMS Alerts")
        print("   📡 Real-time Firebase Monitoring")
        print("   ⏹️  Press Ctrl+C to stop")
        
        # Keep the program running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping real-time monitoring...")
        print("👋 Intelligent AI System shutdown complete")
    except Exception as e:
        print(f"❌ Error in real-time monitoring: {e}")
        import traceback
        traceback.print_exc()