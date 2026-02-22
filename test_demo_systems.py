# D:\sihdeeo\test_demo_system.py
from blockchain_messaging import disease_mapper, language_support, blockchain_utils
from blockchain_messaging.config import USER_PHONE_NUMBERS, USER_LANGUAGES, AUDIO_ALERTS_ENABLED
import time

def test_demo_system():
    print("🎯 DEMO MODE: Blockchain Alert System")
    print("=" * 50)
    print("ℹ️  Running without external APIs")
    print("ℹ️  Audio alerts: " + ("ENABLED" if AUDIO_ALERTS_ENABLED else "DISABLED"))
    print()
    
    # Test water data
    test_water = {
        'pH': 5.8,
        'turbidity': 8.2, 
        'tds': 650,
        'fecal_coliform': 5,
        'location': 'Village Well #1',
        'timestamp': '2024-01-15 14:30:00'
    }
    
    print("📊 Simulated Water Quality Data:")
    for k, v in test_water.items():
        print(f"  {k}: {v}")
    
    # Analyze water quality
    alert_info = disease_mapper.analyze_water_quality(test_water)
    
    if alert_info:
        print(f"\n⚠️  ALERT GENERATED - Risk Level: {alert_info['risk_level']}")
        print(f"   Potential Diseases: {', '.join(alert_info['diseases'])}")
        print(f"   Main Symptoms: {', '.join(alert_info['symptoms'])}")
        print(f"   Prevention: {alert_info['prevention_measures'][0]}")
        
        print("\n" + "=" * 50)
        print("🔊 Testing Multilingual Audio Alerts:")
        
        for phone in USER_PHONE_NUMBERS:
            lang = USER_LANGUAGES.get(phone, "en")
            alert_msg = disease_mapper.generate_alert_message(alert_info, lang)
            
            # Store hash (local demo mode)
            msg_hash = blockchain_utils.generate_hash(alert_msg)
            tx_hash = blockchain_utils.store_on_blockchain(msg_hash)
            
            print(f"\n📨 For: {phone} ({lang.upper()})")
            print(f"   Blockchain TX: {tx_hash}")
            print(f"   Message: {alert_msg}")
            
            # Generate and play audio
            if AUDIO_ALERTS_ENABLED:
                print("   🔊 Playing audio alert...")
                audio_file = language_support.text_to_speech(alert_msg, lang)
                if audio_file:
                    print(f"   ✅ Audio saved: {audio_file}")
                time.sleep(2)  # Wait for audio to play
            else:
                print("   🔇 Audio alerts disabled")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("💡 To enable real SMS alerts, add Twilio credentials to .env file")
    print("💡 To enable blockchain, add Ethereum testnet credentials")

if __name__ == "__main__":
    test_demo_system()