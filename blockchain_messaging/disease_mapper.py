# D:\sihdeeo\blockchain_messaging\disease_mapper.py
import time
from .config import ML_PREDICTION_THRESHOLD, ALERT_COOLDOWN_MINUTES

class DiseaseMapper:
    def __init__(self):
        self.disease_mapping = {
            "high_ph": {
                "diseases": ["Chemical Burns", "Digestive Issues", "Skin Irritation"],
                "symptoms": ["Burning sensation", "Nausea", "Skin rash"],
                "prevention": "DO NOT DRINK! Use bottled water immediately"
            },
            "high_turbidity": {
                "diseases": ["Cholera", "Typhoid", "Diarrhea", "Dysentery"],
                "symptoms": ["Diarrhea", "Vomiting", "Fever", "Dehydration"],
                "prevention": "Boil water for 10+ minutes before use"
            },
            "high_tds": {
                "diseases": ["Kidney stones", "Hypertension", "Digestive problems"],
                "symptoms": ["Abdominal pain", "Frequent urination", "Headache"],
                "prevention": "Use RO water purifier or bottled water"
            },
            "contaminated_water": {
                "diseases": ["Multiple waterborne diseases"],
                "symptoms": ["Gastrointestinal issues", "Fever", "Dehydration"],
                "prevention": "AVOID TAP WATER - Use only purified/bottled water"
            }
        }
        self.last_alert_time = 0
    
    def process_ml_prediction(self, ml_prediction_data):
        """Process ML prediction and generate emergency alerts"""
        try:
            # Check if we should send alert (cooldown period)
            current_time = time.time()
            if current_time - self.last_alert_time < ALERT_COOLDOWN_MINUTES * 60:
                print("⏳ Alert cooldown active - skipping duplicate alert")
                return None
            
            # Extract prediction info
            is_potable = ml_prediction_data.get('prediction', True)
            confidence = ml_prediction_data.get('confidence', 0.0)
            sensor_data = ml_prediction_data.get('sensor_readings', {})
            
            # Only alert for NON-POTABLE water with high confidence
            if is_potable or confidence < ML_PREDICTION_THRESHOLD:
                return None
            
            # Analyze water parameters for specific risks
            alert_info = self._analyze_water_emergency(sensor_data, confidence)
            
            if alert_info:
                self.last_alert_time = current_time
                return alert_info
            
            return None
            
        except Exception as e:
            print(f"Error processing ML prediction: {e}")
            return None
    
    def _analyze_water_emergency(self, sensor_data, confidence):
        """Analyze water parameters for emergency alerts"""
        issues = []
        
        ph = sensor_data.get('pH', 7.0)
        turbidity = sensor_data.get('turbidity', 0)
        tds = sensor_data.get('tds', 0)
        temperature = sensor_data.get('temperature', 25.0)
        
        # CRITICAL ALERTS - Immediate danger
        if ph > 11.0 or ph < 4.0:
            issues.append("chemical_contamination")
        elif ph > 8.5:
            issues.append("high_ph")
        elif ph < 6.5:
            issues.append("low_ph")
        
        if turbidity > 50:
            issues.append("high_turbidity")
        elif turbidity > 10:
            issues.append("moderate_turbidity")
        
        if tds > 1000:
            issues.append("very_high_tds")
        elif tds > 500:
            issues.append("high_tds")
        
        # If multiple issues or extreme values
        if len(issues) >= 2 or "chemical_contamination" in issues:
            issues.append("contaminated_water")
        
        if not issues:
            return None
        
        return self._generate_emergency_alert(issues, confidence, sensor_data)
    
    def _generate_emergency_alert(self, issues, confidence, sensor_data):
        """Generate emergency alert information"""
        alert_info = {
            "risk_level": "CRITICAL" if "chemical_contamination" in issues else "HIGH",
            "issues": issues,
            "diseases": [],
            "symptoms": [], 
            "prevention_measures": [],
            "confidence": confidence,
            "sensor_data": sensor_data,
            "timestamp": time.time(),
            "emergency": True
        }
        
        for issue in issues:
            if issue in self.disease_mapping:
                disease_info = self.disease_mapping[issue]
                alert_info["diseases"].extend(disease_info["diseases"])
                alert_info["symptoms"].extend(disease_info["symptoms"])
                alert_info["prevention_measures"].append(disease_info["prevention"])
        
        # Remove duplicates
        alert_info["diseases"] = list(set(alert_info["diseases"]))
        alert_info["symptoms"] = list(set(alert_info["symptoms"]))
        alert_info["prevention_measures"] = list(set(alert_info["prevention_measures"]))
        
        return alert_info
    
    def generate_emergency_message(self, alert_info, language="en"):
        """Generate URGENT emergency message"""
        if not alert_info:
            return None
        
        sensor_data = alert_info.get('sensor_data', {})
        
        if language == "hi":  # Hindi
            message = "🚨 EMERGENCY - जल संकट 🚨\n"
            message += f"पानी खतरनाक स्तर पर प्रदूषित!\n"
            message += f"pH: {sensor_data.get('pH', 'N/A')} | टर्बिडिटी: {sensor_data.get('turbidity', 'N/A')}\n"
            message += f"जोखिम: {', '.join(alert_info['diseases'][:2])}\n"
            message += f"तुरंत कार्रवाई: {alert_info['prevention_measures'][0]}\n"
            message += "स्वास्थ्य विभाग को सूचित करें!"
        else:  # English
            message = "🚨 EMERGENCY - WATER CRISIS 🚨\n"
            message += f"Water dangerously contaminated!\n"
            message += f"pH: {sensor_data.get('pH', 'N/A')} | Turbidity: {sensor_data.get('turbidity', 'N/A')}\n" 
            message += f"Risks: {', '.join(alert_info['diseases'][:2])}\n"
            message += f"IMMEDIATE ACTION: {alert_info['prevention_measures'][0]}\n"
            message += "Contact health authorities!"
        
        return message

# Singleton instance
disease_mapper = DiseaseMapper()