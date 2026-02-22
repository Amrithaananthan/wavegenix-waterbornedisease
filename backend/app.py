from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model
import os
import json
import sys

# Add current directory to path to fix imports
sys.path.append(os.path.dirname(__file__))

# Import blockchain components
try:
    from blockchain.blockchain import WaterQualityBlockchain, water_blockchain
    from blockchain.alert_manager import AlertManager
    
    # Create alert manager instance
    alert_manager = AlertManager(water_blockchain)
    print("✅ Blockchain modules imported successfully")
    
except ImportError as e:
    print(f"❌ Blockchain import error: {e}")
    print("⚠️ Running without blockchain features")
    # Create mock blockchain classes for fallback
    class MockBlockchain:
        def __init__(self):
            self.chain = []
            self.pending_alerts = []
        def add_alert(self, alert_data): 
            print(f"📝 Mock: Added alert - {alert_data.get('message', 'Unknown')}")
            return alert_data
        def get_alert_history(self): return []
        def get_blockchain_info(self): 
            return {
                'chain_length': 0, 
                'pending_alerts': 0, 
                'is_valid': True, 
                'total_alerts': 0
            }
    
    class MockAlertManager:
        def __init__(self, blockchain): 
            self.blockchain = blockchain
        def create_water_alert(self, sensor_data, ml_prediction, ml_confidence): 
            print(f"📝 Mock: Water alert - Safe: {ml_prediction}, Confidence: {ml_confidence}")
            return {"id": "mock_alert", "message": "Mock alert"}
        def get_alerts_for_display(self, limit=50): 
            return []
        def mine_alerts(self):
            return None
    
    water_blockchain = MockBlockchain()
    alert_manager = MockAlertManager(water_blockchain)

app = Flask(__name__)
CORS(app)

# Firebase setup
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://wavegenix-6-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization error: {e}")
    print("⚠️ Running in offline mode - some features may not work")

# ML Model paths - use relative paths
model_path = os.path.join("models", "water_dl_model.keras")
scaler_path = os.path.join("models", "water_scaler.pkl")

# Load ML model
try:
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ ML Model loaded successfully")
    else:
        print("⚠️ ML model files not found, using mock predictions")
        model = None
        scaler = None
except Exception as e:
    print(f"❌ Error loading ML model: {e}")
    print("⚠️ Running without ML model - predictions will be mock data")
    model = None
    scaler = None

@app.route('/')
def home():
    return jsonify({
        "message": "WaveGenix Backend API",
        "status": "running",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/api/iot-data": "GET - Real-time IoT data",
            "/api/ml-predictions": "GET - ML predictions", 
            "/api/alerts": "GET - Blockchain alerts",
            "/api/historical-data": "GET - Historical data",
            "/api/reports": "POST - Submit reports",
            "/api/blockchain/info": "GET - Blockchain info",
            "/api/health": "GET - Health check"
        }
    })

@app.route('/api/iot-data', methods=['GET'])
def get_iot_data():
    """Get real-time IoT data from Firebase"""
    try:
        # For now, return mock data since Firebase might not be configured
        mock_data = get_mock_iot_data()
        
        # Process with ML
        prediction, confidence = process_with_ml(mock_data)
        
        # Create blockchain alert if water is unsafe
        if prediction is not None and not prediction:
            alert = alert_manager.create_water_alert(mock_data, prediction, confidence)
            print(f"🚨 Blockchain alert created: {alert.get('id', 'unknown')}")
        
        response_data = {
            'pH': mock_data['pH'],
            'tds': mock_data['tds'],
            'turbidity': mock_data['turbidity'],
            'temperature': mock_data['temperature'],
            'deviceId': mock_data['deviceId'],
            'timestamp': mock_data['timestamp'],
            'quality': mock_data['quality'],
            'ml_prediction': bool(prediction) if prediction is not None else None,
            'ml_confidence': confidence
        }
        
        return jsonify(response_data)
            
    except Exception as e:
        print(f"❌ Error in IoT data: {e}")
        return jsonify(get_mock_iot_data())

@app.route('/api/ml-predictions', methods=['GET'])
def get_ml_predictions():
    """Get ML predictions for current data"""
    try:
        return jsonify(get_mock_predictions())
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        return jsonify(get_mock_predictions())

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get blockchain alerts"""
    try:
        limit = request.args.get('limit', 50, type=int)
        alerts = alert_manager.get_alerts_for_display(limit)
        
        # Mine pending alerts every 10 alerts
        if hasattr(water_blockchain, 'pending_alerts') and len(water_blockchain.pending_alerts) >= 5:
            print("⛏️ Mining pending alerts...")
            alert_manager.mine_alerts()
        
        blockchain_info = water_blockchain.get_blockchain_info()
        
        return jsonify({
            'alerts': alerts,
            'blockchain_info': blockchain_info,
            'pending_count': len(water_blockchain.pending_alerts) if hasattr(water_blockchain, 'pending_alerts') else 0
        })
        
    except Exception as e:
        print(f"❌ Error fetching alerts: {e}")
        return jsonify({
            'alerts': [],
            'blockchain_info': {'chain_length': 0, 'pending_alerts': 0, 'is_valid': True, 'total_alerts': 0},
            'pending_count': 0
        })

@app.route('/api/historical-data', methods=['GET'])
def get_historical_data():
    """Get historical data for charts"""
    try:
        days = request.args.get('days', 7, type=int)
        historical_data = generate_historical_data(days)
        return jsonify(historical_data)
    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")
        return jsonify([])

@app.route('/api/reports', methods=['POST'])
def submit_report():
    """Submit water issue reports"""
    try:
        report_data = request.get_json()
        
        print(f"📝 Received report: {report_data}")
        
        # Create blockchain alert for critical reports
        if report_data.get('priority') == 'critical':
            alert_data = {
                'type': 'user_report',
                'severity': 'high',
                'message': f"Critical report from {report_data.get('location', 'Unknown')}: {report_data.get('description', 'No description')}",
                'location': report_data.get('location', 'Unknown'),
                'parameters': {'issue_type': report_data.get('type', 'unknown')}
            }
            alert_manager.blockchain.add_alert(alert_data)
            print("🚨 Critical report alert added to blockchain")
        
        return jsonify({
            'success': True, 
            'message': 'Report submitted successfully',
            'alert_created': report_data.get('priority') == 'critical'
        })
        
    except Exception as e:
        print(f"❌ Error submitting report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/blockchain/info', methods=['GET'])
def get_blockchain_info():
    """Get blockchain information"""
    try:
        info = water_blockchain.get_blockchain_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'services': {
            'blockchain': 'active' if not hasattr(water_blockchain, '__class__') or water_blockchain.__class__.__name__ != 'MockBlockchain' else 'mock',
            'ml_model': 'active' if model else 'mock',
            'firebase': 'active' if 'cred' in locals() else 'offline'
        },
        'blockchain_blocks': len(water_blockchain.chain) if hasattr(water_blockchain, 'chain') else 0,
        'blockchain_alerts': len(water_blockchain.get_alert_history()) if hasattr(water_blockchain, 'get_alert_history') else 0
    })

# Helper functions
def process_with_ml(sensor_data):
    """Process sensor data with ML model"""
    try:
        if model is None or scaler is None:
            return get_mock_prediction()
            
        # Feature mapping
        feature_mapping = {
            'Ph': sensor_data.get('pH', 7.0),
            'Turbidity': sensor_data.get('turbidity', 0),
            'Temperature': sensor_data.get('temperature', 25.0),
            'Conductivity': sensor_data.get('tds', 0)
        }
        
        # Prepare features
        features = ['Ph', 'Turbidity', 'Temperature', 'Conductivity']
        X = np.array([[float(feature_mapping[feat]) for feat in features]])
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0)[0][0]
        
        return pred > 0.5, float(pred)
        
    except Exception as e:
        print(f"❌ ML processing error: {e}")
        return get_mock_prediction()

def get_mock_prediction():
    """Generate mock prediction when ML is unavailable"""
    is_safe = np.random.random() > 0.2  # 80% chance of safe water
    confidence = 0.85 + np.random.random() * 0.1 if is_safe else 0.7 + np.random.random() * 0.2
    return is_safe, confidence

def get_mock_predictions():
    """Generate complete mock predictions"""
    is_safe = np.random.random() > 0.2
    confidence = 0.85 + np.random.random() * 0.1 if is_safe else 0.7 + np.random.random() * 0.2
    
    if is_safe:
        analysis = ["✅ pH within safe range", "✅ TDS within safe range", "✅ Turbidity within safe range"]
        recommendation = "Water quality is good. No immediate action needed."
    else:
        analysis = ["🚨 pH too acidic", "⚠️ High TDS levels", "✅ Turbidity within safe range"]
        recommendation = "🚨 WATER UNSAFE! Do not drink. Contact authorities immediately."
    
    return {
        'prediction': int(is_safe),
        'confidence': float(confidence),
        'safety_analysis': analysis,
        'recommendation': recommendation,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'pH': 7.0 + np.random.normal(0, 0.5),
            'tds': 300 + np.random.normal(0, 100),
            'turbidity': 1.0 + np.random.random(),
            'temperature': 25 + np.random.normal(0, 3)
        }
    }

def get_mock_iot_data():
    """Generate mock IoT data"""
    return {
        'pH': 7.2 + (np.random.random() - 0.5) * 0.4,
        'tds': 320 + (np.random.random() - 0.5) * 40,
        'temperature': 24 + (np.random.random() - 0.5) * 2,
        'turbidity': 0.8 + np.random.random() * 0.4,
        'timestamp': datetime.now().isoformat(),
        'deviceId': "Mock_Device_001",
        'quality': "Good",
        'location': "Test Village"
    }

def generate_historical_data(days):
    """Generate mock historical data"""
    data = []
    now = datetime.now()
    
    for i in range(days, -1, -1):
        date = now - timedelta(days=i)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'pH': 7.0 + np.random.normal(0, 0.3),
            'tds': 300 + np.random.normal(0, 50),
            'temperature': 25 + np.random.normal(0, 2),
            'turbidity': 1.0 + np.random.normal(0, 0.5),
            'quality': 'Good' if np.random.random() > 0.3 else 'Moderate'
        })
    
    return data

if __name__ == '__main__':
    print("🚀 Starting WaveGenix Backend Server...")
    print("🔗 Blockchain system: " + ("ACTIVE" if not 'MockBlockchain' in str(type(water_blockchain)) else "MOCK MODE"))
    print("🤖 ML Model: " + ("LOADED" if model else "MOCK MODE"))
    print("📡 Firebase: " + ("CONNECTED" if 'cred' in locals() else "OFFLINE MODE"))
    print("🌐 Server running on: http://localhost:5000")
    print("📊 Available endpoints:")
    print("   GET  /")
    print("   GET  /api/iot-data")
    print("   GET  /api/ml-predictions") 
    print("   GET  /api/alerts")
    print("   GET  /api/historical-data")
    print("   POST /api/reports")
    print("   GET  /api/blockchain/info")
    print("   GET  /api/health")
    
    # Add some initial test alerts
    try:
        # Add a test alert to blockchain
        test_alert = {
            'type': 'info',
            'severity': 'low', 
            'message': 'WaveGenix system initialized successfully',
            'location': 'System',
            'parameters': {'status': 'online'}
        }
        water_blockchain.add_alert(test_alert)
        print("✅ Test alert added to blockchain")
    except Exception as e:
        print(f"⚠️ Could not add test alert: {e}")
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"❌ Failed to start server: {e}")