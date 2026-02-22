# quick_test.py
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://wavegenix-6-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Test connection
ref = db.reference('/')
data = ref.get()

print("✅ Firebase Connection Successful!")
print(f"📊 Data type: {type(data)}")
if data:
    print(f"🔍 Top-level keys: {list(data.keys())}")
    print(f"📝 Sample record: {list(data.values())[0] if data else 'No data'}")
else:
    print("❌ No data found at root level")