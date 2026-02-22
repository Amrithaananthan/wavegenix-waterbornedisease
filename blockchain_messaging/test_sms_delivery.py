# test_sms_delivery.py
from message_sender import message_sender

# Simple test message
test_msg = "TEST: Water Alert System - Please reply if received"
success = message_sender.send_emergency_alert(test_msg)

if success:
    print("✅ Test message sent - check phone in 1-2 minutes")
else:
    print("❌ Test failed - check Twilio console for errors")