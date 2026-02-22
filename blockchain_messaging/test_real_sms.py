# D:\newsih\sihdeeo\blockchain_messaging\test_real_sms.py
from message_sender import message_sender

def test_emergency_alert():
    print("🚀 TESTING WATER CRISIS ALERT SYSTEM")
    print("=" * 50)
    
    # Real emergency message
    emergency_msg = """🚨 EMERGENCY - WATER CRISIS 🚨

CONTAMINATION DETECTED:
• pH: 13.1 (EXTREMELY ALKALINE)
• Turbidity: 73.5 NTU (VERY MURKY)
• TDS: 608 ppm (HIGH MINERALS)

🚫 DO NOT DRINK THIS WATER!
✅ USE BOTTLED WATER ONLY
🏥 CONTACT HEALTH AUTHORITIES

AI Water Monitoring System"""

    print("Sending REAL emergency alert...")
    success = message_sender.send_emergency_alert(emergency_msg)
    
    if success:
        print("🎉 REAL SMS ALERT SENT SUCCESSFULLY!")
        print("📱 Check your phone for the emergency message!")
    else:
        print("❌ SMS failed. Please check:")
        print("   1. Auth Token in message_sender.py")
        print("   2. Phone number verification in Twilio")
        print("   3. Trial account balance")

if __name__ == "__main__":
    test_emergency_alert()