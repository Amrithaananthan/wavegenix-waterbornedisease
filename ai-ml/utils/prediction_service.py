import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json
from utils.database_utils import db_connector
import warnings
warnings.filterwarnings('ignore')

class PredictionService:
    def __init__(self):
        self.model = None
        self.model_metadata = None
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the most recent trained model"""
        try:
            import glob
            import os
            
            # Find the latest model file
            model_files = glob.glob('../models/outbreak_model_*.joblib')
            if not model_files:
                print("No trained models found. Please train a model first.")
                return False
            
            # Get the latest model by timestamp
            latest_model = max(model_files, key=os.path.getctime)
            
            # Load model
            self.model = joblib.load(latest_model)
            
            # Load corresponding metadata
            model_id = latest_model.split('_')[-1].split('.')[0]
            metadata_file = f'../models/model_metadata_{model_id}.json'
            
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
            
            print(f"✅ Loaded model: {latest_model}")
            print(f"Model type: {self.model_metadata['model_type']}")
            print(f"Trained on: {self.model_metadata['training_date']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, sensor_data):
        """Predict outbreak risk for sensor data"""
        if self.model is None:
            return {
                'error': 'No trained model available',
                'risk_level': 'unknown',
                'confidence': 0.0,
                'risk_score': 0
            }
        
        try:
            # Prepare data for prediction
            df = pd.DataFrame([sensor_data])
            
            # Extract features
            features = self.model_metadata['features']
            available_features = [f for f in features if f in df.columns]
            
            # Add time-based features if timestamp is provided
            if 'timestamp' in df.columns and 'hour' not in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
            
            X = df[available_features]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Prepare result
            result = {
                'risk_level': 'high_risk' if prediction == 1 else 'low_risk',
                'confidence': float(probability[1] if prediction == 1 else probability[0]),
                'risk_score': float(probability[1] * 10),  # Scale to 0-10
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_metadata['timestamp'],
                'features_used': available_features
            }
            
            # Save prediction to database
            db_connector.save_prediction(result)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'risk_level': 'error',
                'confidence': 0.0,
                'risk_score': 0
            }
    
    def get_risk_factors(self, sensor_data):
        """Analyze which factors contribute most to risk"""
        if self.model is None or 'feature_importance' not in self.model_metadata:
            return []
        
        try:
            feature_importance = self.model_metadata['feature_importance']
            factors = []
            
            for feature, importance in feature_importance.items():
                if feature in sensor_data:
                    value = sensor_data[feature]
                    threshold = self.get_threshold(feature)
                    
                    factor = {
                        'feature': feature,
                        'value': value,
                        'importance': importance,
                        'status': 'normal'
                    }
                    
                    # Determine if value is concerning
                    if feature == 'pH' and (value < 6.5 or value > 8.5):
                        factor['status'] = 'concerning'
                    elif feature == 'turbidity' and value > 10:
                        factor['status'] = 'concerning'
                    elif feature == 'tds' and value > 500:
                        factor['status'] = 'concerning'
                    elif feature == 'temperature' and (value > 35 or value < 10):
                        factor['status'] = 'concerning'
                    
                    factors.append(factor)
            
            # Sort by importance
            factors.sort(key=lambda x: x['importance'], reverse=True)
            return factors
            
        except Exception as e:
            print(f"Error analyzing risk factors: {e}")
            return []
    
    def get_threshold(self, feature):
        """Get safety thresholds for different features"""
        thresholds = {
            'pH': (6.5, 8.5),
            'turbidity': (0, 10),
            'tds': (0, 500),
            'temperature': (10, 35)
        }
        return thresholds.get(feature, (None, None))
    
    def batch_predict(self, sensor_data_list):
        """Predict for multiple data points"""
        results = []
        for data in sensor_data_list:
            results.append(self.predict(data))
        return results

# Global instance
prediction_service = PredictionService()

# Test function
def test_prediction():
    print("Testing prediction service...")
    
    # Sample data
    sample_data = {
        'pH': 6.8,
        'turbidity': 18.2,
        'tds': 620,
        'temperature': 29.1,
        'timestamp': datetime.now().isoformat(),
        'deviceId': 'test_device_1',
        'location': {'lat': 28.6139, 'lng': 77.2090}
    }
    
    # Make prediction
    result = prediction_service.predict(sample_data)
    print("Prediction Result:", json.dumps(result, indent=2))
    
    # Analyze risk factors
    factors = prediction_service.get_risk_factors(sample_data)
    print("\nRisk Factors Analysis:")
    for factor in factors:
        print(f"  {factor['feature']}: {factor['value']} (importance: {factor['importance']:.3f}, status: {factor['status']})")

if __name__ == "__main__":
    test_prediction()
