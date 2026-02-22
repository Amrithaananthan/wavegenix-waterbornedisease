import requests
from datetime import datetime

class AlertManager:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        
    def create_water_alert(self, sensor_data, ml_prediction, ml_confidence):
        """Create blockchain alert based on ML predictions"""
        
        alert_type = "critical" if not ml_prediction else "warning"
        severity = "high" if ml_confidence > 0.8 else "medium"
        
        alert_data = {
            'type': alert_type,
            'severity': severity,
            'message': self._generate_alert_message(sensor_data, ml_prediction, ml_confidence),
            'parameters': {
                'pH': sensor_data.get('pH', 0),
                'tds': sensor_data.get('tds', 0),
                'turbidity': sensor_data.get('turbidity', 0),
                'temperature': sensor_data.get('temperature', 0)
            },
            'location': sensor_data.get('location', 'Unknown'),
            'device_id': sensor_data.get('deviceId', 'Unknown'),
            'ml_confidence': ml_confidence,
            'prediction': 'unsafe' if not ml_prediction else 'safe'
        }
        
        # Add to blockchain
        blockchain_alert = self.blockchain.add_alert(alert_data)
        
        # Trigger immediate notifications for critical alerts
        if alert_type == 'critical':
            self._trigger_emergency_notifications(blockchain_alert)
        
        return blockchain_alert
    
    def _generate_alert_message(self, sensor_data, prediction, confidence):
        if not prediction:
            return f"🚨 UNSAFE WATER DETECTED! pH: {sensor_data.get('pH', 'N/A')}, " \
                   f"TDS: {sensor_data.get('tds', 'N/A')}ppm, Confidence: {confidence:.1%}"
        else:
            return f"⚠️ Water quality warning. Parameters approaching unsafe levels. " \
                   f"Confidence: {confidence:.1%}"
    
    def _trigger_emergency_notifications(self, alert):
        """Send emergency notifications via SMS/Email"""
        try:
            # SMS Notification - You can integrate with Twilio or other SMS services
            print(f"📱 EMERGENCY SMS WOULD BE SENT: {alert['message']}")
            print("💡 Integrate with Twilio API for actual SMS sending")
                
        except Exception as e:
            print(f"❌ Notification error: {e}")
    
    def mine_alerts(self):
        """Mine pending alerts into blockchain"""
        return self.blockchain.mine_pending_alerts()
    
    def get_alerts_for_display(self, limit=50):
        """Get formatted alerts for frontend display"""
        all_alerts = self.blockchain.get_alert_history()
        recent_alerts = sorted(all_alerts, 
                             key=lambda x: x['timestamp'], 
                             reverse=True)[:limit]
        
        formatted_alerts = []
        for alert in recent_alerts:
            formatted_alerts.append({
                'id': alert['id'],
                'type': alert['type'],
                'severity': alert['severity'],
                'message': alert['message'],
                'timestamp': alert['timestamp'],
                'location': alert.get('location', 'Unknown'),
                'parameters': alert.get('parameters', {}),
                'ml_confidence': alert.get('ml_confidence', 0)
            })
        
        return formatted_alerts

# Don't create alert_manager instance here - it will be created in app.py