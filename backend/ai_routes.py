from flask import Blueprint, request, jsonify
from datetime import datetime
import json
import sys
import os

# Add AI/ML directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai-ml'))

from ai_ml.utils.prediction_service import prediction_service

# Create Blueprint for AI routes
ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/api/ai/predict', methods=['POST'])
def predict_outbreak():
    """Predict outbreak risk for sensor data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = prediction_service.predict(data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/ai/analyze-risk', methods=['POST'])
def analyze_risk_factors():
    """Analyze risk factors for sensor data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Analyze risk factors
        factors = prediction_service.get_risk_factors(data)
        
        return jsonify({
            'risk_factors': factors,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/ai/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    try:
        if prediction_service.model_metadata:
            return jsonify(prediction_service.model_metadata), 200
        else:
            return jsonify({'error': 'No model loaded'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/ai/retrain', methods=['POST'])
def retrain_model():
    """Retrain the AI model with latest data"""
    try:
        from ai_ml.training.train_model import OutbreakPredictor
        
        days = request.args.get('days', 90, type=int)
        
        predictor = OutbreakPredictor()
        success = predictor.train_models(days=days)
        
        if success:
            # Reload the new model
            prediction_service.load_latest_model()
            
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully',
                'days_of_data': days
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_bp.route('/api/ai/predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions from database"""
    try:
        predictions = prediction_service.get_recent_predictions(limit=20)
        
        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            
        return jsonify(predictions), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
