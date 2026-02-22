from flask import Flask, jsonify, request
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime, timedelta
import json
import os

# Disable Flask dotenv loading completely
os.environ['FLASK_SKIP_DOTENV'] = '1'

# Create Flask app without dotenv
app = Flask(__name__)
CORS(app)

print("🚀 Starting WaveGenix Backend Server...")

# Initialize Firebase
try:
    cred_path = "serviceAccountKey.json"
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://wavegenix-6-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("✅ Firebase initialized successfully")
    else:
        print(f"❌ Firebase credentials file not found at: {cred_path}")
        print("💡 Please ensure serviceAccountKey.json is in the same directory")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")

def is_real_iot_data(data):
    """Check if data is real IoT sensor data with more flexible structure"""
    if not isinstance(data, dict):
        return False
    
    # Check for nested waterData structure
    if 'waterData' in data and isinstance(data['waterData'], dict):
        water_data = data['waterData']
        required_fields = ['pH', 'tds', 'turbidity', 'temperature']
        return any(field in water_data for field in required_fields)
    
    # Check for direct fields
    required_fields = ['pH', 'tds', 'turbidity', 'temperature']
    return any(field in data for field in required_fields)

@app.route('/')
def home():
    return jsonify({
        "message": "WaveGenix API Server", 
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/iot-data')
def get_iot_data():
    """Get latest IoT sensor data from Firebase"""
    try:
        ref = db.reference('/')
        snapshot = ref.get()
        
        if not snapshot:
            print("📭 No data found in Firebase database")
            return jsonify(get_fallback_iot_data())
        
        latest_data = None
        for key, data in snapshot.items():
            if key != 'ml_predictions' and is_real_iot_data(data):
                latest_data = data
                print(f"📊 Found IoT data: {key} - {data}")
                break
        
        if latest_data:
            # Extract data from nested structure or direct fields
            if 'waterData' in latest_data:
                water_data = latest_data['waterData']
            else:
                water_data = latest_data
            
            response_data = {
                "pH": float(water_data.get('pH', water_data.get('Ph', 7.0))),
                "tds": float(water_data.get('tds', water_data.get('TDS', water_data.get('Conductivity', 300)))),
                "turbidity": float(water_data.get('turbidity', water_data.get('Turbidity', 1.0))),
                "temperature": float(water_data.get('temperature', water_data.get('Temperature', 25.0))),
                "deviceId": water_data.get('deviceId', water_data.get('deviceID', 'Unknown')),
                "timestamp": water_data.get('timestamp', datetime.now().isoformat()),
                "quality": water_data.get('qualityScore', water_data.get('quality', 'Good'))
            }
            print(f"📈 Sending REAL IoT data: pH={response_data['pH']}, TDS={response_data['tds']}")
            return jsonify(response_data)
        else:
            print("📭 No real IoT data found, using fallback")
            return jsonify(get_fallback_iot_data())
            
    except Exception as e:
        print(f"❌ Error fetching IoT data: {e}")
        return jsonify(get_fallback_iot_data())

def get_fallback_iot_data():
    """Get fallback IoT data that's clearly marked as test data"""
    return {
        "pH": 6.8,  # Different from frontend default
        "tds": 450, # Different from frontend default  
        "turbidity": 1.2,
        "temperature": 26.0,
        "deviceId": "TEST_Fallback_Device",
        "timestamp": datetime.now().isoformat(),
        "quality": "Moderate"
    }

@app.route('/api/ml-predictions')
def get_ml_predictions():
    """Get ML predictions"""
    try:
        ref = db.reference('/ml_predictions')
        predictions = ref.get()
        
        if predictions:
            if isinstance(predictions, dict):
                latest_key = list(predictions.keys())[-1]
                latest_pred = predictions[latest_key]
            else:
                latest_pred = predictions
            
            print(f"🤖 Found ML prediction: {latest_pred}")
                
            return jsonify({
                "prediction": latest_pred.get('prediction', 1),
                "confidence": latest_pred.get('confidence', 0.85),
                "safety_analysis": latest_pred.get('safety_analysis', [
                    "✅ pH within safe range",
                    "✅ TDS within safe range", 
                    "✅ Turbidity within safe range"
                ]),
                "recommendation": latest_pred.get('recommendation', "Water quality is normal"),
                "timestamp": latest_pred.get('timestamp', datetime.now().isoformat())
            })
        else:
            print("🤖 No ML predictions found, using fallback")
            return jsonify(get_fallback_ml_predictions())
            
    except Exception as e:
        print(f"❌ Error fetching ML predictions: {e}")
        return jsonify(get_fallback_ml_predictions())

def get_fallback_ml_predictions():
    """Get fallback ML predictions"""
    return {
        "prediction": 1,
        "confidence": 0.92,
        "safety_analysis": [
            "✅ pH within safe range (6.5-8.5)",
            "✅ TDS within safe range (<500 ppm)",
            "✅ Turbidity within safe range (<5 NTU)"
        ],
        "recommendation": "Water quality is good. No immediate action needed.",
        "timestamp": datetime.now().isoformat()
    }

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical data for charts"""
    try:
        days = request.args.get('days', 7, type=int)
        historical_data = generate_historical_data(days)
        print(f"📊 Generated {len(historical_data)} days of historical data")
        return jsonify(historical_data)
    except Exception as e:
        print(f"❌ Error generating historical data: {e}")
        return jsonify([])

def generate_historical_data(days):
    """Generate historical data"""
    data = []
    base_date = datetime.now()
    
    for i in range(days, -1, -1):
        date = base_date - timedelta(days=i)
        
        base_ph = 7.0 + (i % 7) * 0.1
        base_tds = 300 + (i % 5) * 40
        base_temp = 24 + (i % 3)
        base_turbidity = 0.8 + (i % 2) * 0.3
        
        if base_ph >= 6.5 and base_ph <= 8.5 and base_tds <= 500:
            quality = "Good"
        elif base_ph >= 6.0 and base_ph <= 9.0 and base_tds <= 800:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        data.append({
            "date": date.strftime('%Y-%m-%d'),
            "pH": round(base_ph, 1),
            "tds": base_tds,
            "temperature": base_temp,
            "turbidity": round(base_turbidity, 1),
            "quality": quality
        })
    
    return data

@app.route('/api/reports', methods=['POST'])
def submit_report():
    """Submit water quality reports"""
    try:
        report_data = request.json
        print(f"📝 Received report: {report_data.get('type', 'Unknown')} - {report_data.get('location', 'Unknown')}")
        
        # Store in Firebase
        ref = db.reference('/reports')
        new_report_ref = ref.push()
        new_report_ref.set({
            **report_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'id': new_report_ref.key
        })
        
        print(f"✅ Report stored with ID: {new_report_ref.key}")
        
        return jsonify({
            "success": True,
            "reportId": new_report_ref.key,
            "message": "Report submitted successfully"
        })
        
    except Exception as e:
        print(f"❌ Error submitting report: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/debug-firebase')
def debug_firebase():
    """Debug endpoint to check Firebase structure"""
    try:
        ref = db.reference('/')
        snapshot = ref.get()
        
        return jsonify({
            "firebase_data": snapshot,
            "message": "Firebase connection successful" if snapshot else "No data in Firebase"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Firebase connection failed"
        }), 500

if __name__ == '__main__':
    print("📡 Available Endpoints:")
    print("   http://localhost:5000/")
    print("   http://localhost:5000/api/iot-data")
    print("   http://localhost:5000/api/ml-predictions")
    print("   http://localhost:5000/api/historical-data")
    print("   http://localhost:5000/api/reports")
    print("   http://localhost:5000/api/health")
    print("   http://localhost:5000/api/debug-firebase")
    print("\n🌐 Server ready!")
    
    # Run without debug mode to avoid reloader issues
    app.run(host='0.0.0.0', port=5000, debug=False)