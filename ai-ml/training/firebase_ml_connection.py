import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Initialize Firebase with your service account key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://wavegenix-6-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# 2. Function to get all data from database
def get_all_firebase_data():
    try:
        ref = db.reference('/')  # Root of database
        snapshot = ref.get()
        
        if snapshot:
            print("✅ Successfully connected to Firebase!")
            print(f"📊 Found {len(snapshot)} top-level records")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame.from_dict(snapshot, orient='index')
            return df
        else:
            print("❌ No data found in database")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return pd.DataFrame()

# 3. Real-time data listener for live ML predictions
def real_time_ml_listener(event):
    print(f"\n🔄 New data received at {datetime.now()}:")
    print(f"Path: {event.path}")
    print(f"Data: {event.data}")
    
    # Your ML processing here
    process_with_ml(event.data)

# 4. ML Processing Function (Template)
def process_with_ml(data):
    """Process new data with your ML model"""
    try:
        # Example: Extract features and make prediction
        features = extract_features(data)
        
        # Your ML model prediction (example)
        # prediction = your_ml_model.predict([features])
        prediction = 0.85  # Placeholder
        
        print(f"🤖 ML Prediction: {prediction}")
        
        # Store prediction back to Firebase if needed
        store_prediction(data, prediction)
        
    except Exception as e:
        print(f"❌ ML processing error: {e}")

def extract_features(data):
    """Extract features from your IoT data"""
    # Customize this based on your data structure
    if isinstance(data, dict):
        return [float(data.get('value', 0))]  # Example feature extraction
    return [0.0]

def store_prediction(original_data, prediction):
    """Store ML predictions back to Firebase"""
    try:
        ref = db.reference('/predictions')
        new_prediction_ref = ref.push()
        new_prediction_ref.set({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'original_data': original_data
        })
        print("✅ Prediction stored in Firebase")
    except Exception as e:
        print(f"❌ Error storing prediction: {e}")

# 5. Main execution
if __name__ == "__main__":
    print("🚀 Starting Firebase ML Connection...")
    
    # Get all historical data for ML training
    print("\n📖 Loading historical data...")
    historical_data = get_all_firebase_data()
    
    if not historical_data.empty:
        print("\n📈 Historical Data Preview:")
        print(historical_data.head())
        print(f"\n📊 Data Shape: {historical_data.shape}")
        print(f"\n📋 Columns: {historical_data.columns.tolist()}")
        
        # TODO: Train your ML model here
        # model = train_ml_model(historical_data)
        print("🤖 ML model training ready!")
    
    # Start real-time listening (uncomment when ready)
    print("\n🎧 Setting up real-time listener...")
    # Replace '/your-data-path' with your actual data path
    # ref = db.reference('/sensors')  # Change this path!
    # ref.listen(real_time_ml_listener)
    
    print("\n✅ Firebase ML connection setup complete!")
    print("💡 Don't forget to:")
    print("   1. Check your actual data path in Firebase Console")
    print("   2. Uncomment real-time listener with correct path")
    print("   3. Implement your actual ML model logic")