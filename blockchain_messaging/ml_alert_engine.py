# D:\sihdeeo\blockchain_messaging\ml_alert_engine.py
import json
import time
import os
from .disease_mapper import disease_mapper
from .message_sender import message_sender
from .language_support import language_support
from .blockchain_utils import blockchain_utils
from .config import USER_PHONE_NUMBERS, USER_LANGUAGES, AUDIO_ALERTS_ENABLED

class MLAlertEngine:
    def __init__(self):
        print("🚀 ML Alert Engine Initialized - Ready for emergency alerts")
    
    def process_ml_prediction(self, ml_prediction_data):
        """Main function: Process ML prediction and trigger emergency alerts"""
        try:
            print(f"🔍 Processing ML prediction: {ml_prediction_data.get('prediction')}")
            
            # Analyze water quality for emergencies
            alert_info = disease_mapper.process_ml_prediction(ml_prediction_data)
            
            if not alert_info:
                print("✅ Water is safe - no alert needed")
                return False
            
            print(f"🚨 EMERGENCY DETECTED: {alert_info['risk_level']} risk")
            
            # Send alerts to all users
            self._send_emergency_alerts(alert_info)
            
            # Store on blockchain for transparency
            self._store_alert_on_blockchain(alert_info)
            
            # Play audio alert if enabled
            if AUDIO_ALERTS_ENABLED:
                self._play_audio_alert(alert_info)
            
            print("✅ Emergency alert process completed")
            return True
            
        except Exception as e:
            print(f"❌ Alert processing failed: {e}")
            return False
    
    def _send_emergency_alerts(self, alert_info):
        """Send emergency alerts to all registered users"""
        print("📤 Sending emergency alerts to all users...")
        
        for phone_number in USER_PHONE_NUMBERS:
            language = USER_LANGUAGES.get(phone_number, "en")
            alert_message = disease_mapper.generate_emergency_message(alert_info, language)
            
            if alert_message:
                print(f"📱 Preparing alert for {phone_number} in {language}")
                message_sender.send_emergency_alert(alert_message)
        
        print("✅ All emergency alerts dispatched")
    
    def _store_alert_on_blockchain(self, alert_info):
        """Store alert hash on blockchain for transparency"""
        try:
            alert_data = {
                'risk_level': alert_info['risk_level'],
                'issues': alert_info['issues'],
                'timestamp': alert_info['timestamp'],
                'sensor_data': alert_info.get('sensor_data', {})
            }
            
            alert_json = json.dumps(alert_data, sort_keys=True)
            message_hash = blockchain_utils.generate_hash(alert_json)
            tx_hash = blockchain_utils.store_on_blockchain(message_hash)
            
            print(f"⛓️ Alert stored on blockchain: {tx_hash}")
            
        except Exception as e:
            print(f"❌ Blockchain storage failed: {e}")
    
    def _play_audio_alert(self, alert_info):
        """Play audio alert in different languages"""
        try:
            # Play in Hindi
            hindi_message = disease_mapper.generate_emergency_message(alert_info, "hi")
            if hindi_message:
                language_support.text_to_speech(hindi_message, "hi", True)
                time.sleep(2)
            
            # Play in English  
            english_message = disease_mapper.generate_emergency_message(alert_info, "en")
            if english_message:
                language_support.text_to_speech(english_message, "en", True)
                
        except Exception as e:
            print(f"❌ Audio alert failed: {e}")

# Singleton instance
ml_alert_engine = MLAlertEngine()