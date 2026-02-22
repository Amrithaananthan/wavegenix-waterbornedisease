# free_sms_sender.py
import requests
import json

class FreeSMSSender:
    def __init__(self):
        self.services = [
            self._fast2sms,    # Indian SMS service
            self._textbelt,    # International (1 free/day)
            self._callmebot,   # WhatsApp/Telegram
        ]
    
    def send_emergency_alert(self, message):
        """Try multiple free services until one works"""
        for service in self.services:
            print(f"🔄 Trying {service.__name__}...")
            success = service(message)
            if success:
                return True
        return False
    
    def _fast2sms(self, message):
        """Fast2SMS - Free for limited messages"""
        try:
            # You need to sign up at fast2sms.com (free)
            url = "https://www.fast2sms.com/dev/bulkV2"
            
            # Simplified message for Indian carriers
            clean_message = message.replace("🚨", "ALERT").replace("💧", "WATER")[:140]
            
            payload = {
                "message": clean_message,
                "language": "english",
                "route": "q",
                "numbers": "918098216997"  # Your number without +
            }
            
            headers = {
                'authorization': 'YOUR_FREE_API_KEY',  # Get from fast2sms.com
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                print("✅ Fast2SMS: Message sent!")
                return True
        except Exception as e:
            print(f"❌ Fast2SMS failed: {e}")
        return False
    
    def _textbelt(self, message):
        """TextBelt - 1 free SMS per day"""
        try:
            resp = requests.post('https://textbelt.com/text', {
                'phone': '+918098216997',
                'message': message[:160],
                'key': 'textbelt',  # Free key
            })
            
            result = resp.json()
            if result.get('success'):
                print("✅ TextBelt: Free SMS sent! (1/day limit)")
                return True
            else:
                print(f"❌ TextBelt: {result.get('error')}")
        except Exception as e:
            print(f"❌ TextBelt failed: {e}")
        return False
    
    def _callmebot(self, message):
        """CallMeBot - Free WhatsApp/Telegram messages"""
        try:
            # WhatsApp API (free)
            whatsapp_url = f"https://api.callmebot.com/whatsapp.php?phone=+918098216997&text={message}&apikey=YOUR_KEY"
            
            # Telegram API (free)
            telegram_url = f"https://api.callmebot.com/text.php?user=@YOUR_TELEGRAM_USER&text={message}"
            
            response = requests.get(whatsapp_url)
            if response.status_code == 200:
                print("✅ CallMeBot: WhatsApp message sent!")
                return True
        except Exception as e:
            print(f"❌ CallMeBot failed: {e}")
        return False

# Global instance
free_sms_sender = FreeSMSSender()