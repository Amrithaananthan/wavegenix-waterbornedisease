# D:\sihdeeo\test_alert_system.py
from blockchain_messaging import disease_mapper, language_support, blockchain_utils, message_sender
from blockchain_messaging.config import USER_PHONE_NUMBERS, USER_LANGUAGES
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_alert_system():
    print("🚀 Testing Improved Alert System")
    print("=" * 50)
    
    # Test water data
    test_water = {
        'pH': 5.8,
        'turbidity': 8.2,
        'tds': 650,
        'fecal_coliform': 5
    }
    
    print("📊 Water Quality Analysis:")
    for k, v in test_water.items():
        print(f"{k}: {v}")
    
    # Analyze and generate alert
    alert_info = disease_mapper.analyze_water_quality(test_water)
    
    if alert_info:
        print(f"\n⚠️  ALERT - Risk: {alert_info['risk_level']}")
        print(f"Diseases: {', '.join(alert_info['diseases'])}")
        
        for phone in USER_PHONE_NUMBERS:
            lang = USER_LANGUAGES.get(phone, "en")
            alert_msg = disease_mapper.generate_alert_message(alert_info, lang)
            
            # Store on blockchain
            msg_hash = blockchain_utils.generate_hash(alert_msg)
            tx_hash = blockchain_utils.store_on_blockchain(msg_hash)
            
            print(f"\n📨 To {phone} ({lang}):")
            print(f"TX: {tx_hash}")
            print(f"Message: {alert_msg}")
            
            # Send SMS
            success = message_sender.send_sms(phone, alert_msg)
            print(f"Status: {'✅ Success' if success else '❌ Failed (saved to outbox)'}")
            
            # Generate audio
            audio_file = language_support.text_to_speech(alert_msg, lang)
            if audio_file:
                print(f"🎵 Audio: {audio_file}")

if __name__ == "__main__":
    test_alert_system()