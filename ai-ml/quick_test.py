import pandas as pd
import joblib
import os
from datetime import datetime

def quick_test():
    """Quick test without JSON dependencies"""
    print("🚀 Quick Water Quality Test")
    print("===========================")
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
    
    # Find the latest model
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
    if not model_files:
        print("❌ No model found. Please train the model first.")
        return
    
    latest_model = max(model_files, key=lambda f: os.path.getctime(os.path.join(MODEL_DIR, f)))
    model_path = os.path.join(MODEL_DIR, latest_model)
    
    print(f"📂 Using model: {latest_model}")
    model = joblib.load(model_path)
    
    # Test samples
    samples = [
        {
            "ph": 7.2, "Hardness": 180, "Solids": 20000, "Chloramines": 7.5,
            "Sulfate": 350, "Conductivity": 450, "Organic_carbon": 15,
            "Trihalomethanes": 80, "Turbidity": 4.0
        },
        {
            "ph": 5.5, "Hardness": 280, "Solids": 50000, "Chloramines": 8.5,
            "Sulfate": 450, "Conductivity": 650, "Organic_carbon": 25,
            "Trihalomethanes": 110, "Turbidity": 8.0
        }
    ]
    
    for i, sample in enumerate(samples):
        print(f"\n💧 Sample {i+1} Analysis:")
        print("-" * 30)
        
        df = pd.DataFrame([sample])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][prediction]
        
        print(f"🔍 Prediction: {'POTABLE 💧' if prediction == 1 else 'NOT POTABLE ⚠️'}")
        print(f"📊 Confidence: {probability:.3f}")
        print(f"🦠 Disease Risk: {'HIGH' if prediction == 0 else 'LOW'}")
        
        # Blockchain message
        blockchain_msg = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": i+1,
            "status": "SAFE" if prediction == 1 else "UNSAFE",
            "actions": ["OK to drink" if prediction == 1 else "Boil before use"]
        }
        print(f"📡 Blockchain: {blockchain_msg}")

if __name__ == "__main__":
    quick_test()