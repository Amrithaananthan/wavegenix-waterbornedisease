import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient
from datetime import datetime, timedelta

class DatabaseConnector:
    def __init__(self):
        self.mongo_uri = "mongodb://localhost:27017/aquaguard"
        self.sql_uri = "sqlite:///../backend/data.db"
        
    def connect_mongo(self):
        """Connect to MongoDB database"""
        try:
            client = MongoClient(self.mongo_uri)
            db = client["aquaguard"]
            return db
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return None
    
    def connect_sql(self):
        """Connect to SQL database"""
        try:
            engine = create_engine(self.sql_uri)
            return engine
        except Exception as e:
            print(f"SQL connection error: {e}")
            return None
    
    def fetch_training_data(self, days=30):
        """Fetch historical data for training"""
        try:
            db = self.connect_mongo()
            if db is not None:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Fetch data from MongoDB
                collection = db["iot_data"]
                data = list(collection.find({
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }))
                
                if not data:
                    print("⚠️ No training data found in MongoDB for given date range")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Clean and preprocess
                df = self.preprocess_data(df)
                return df
            else:
                print("⚠️ MongoDB connection not available")
                return None
        except Exception as e:
            print(f"Error fetching training data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        if df.empty:
            return df
        
        # Convert timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Extract features
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month
        
        # Handle missing values
        numeric_cols = ["pH", "turbidity", "tds", "temperature"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col].fillna(df[col].median(), inplace=True)
        
        # Add target variable (outbreak risk)
        df["outbreak_risk"] = self.calculate_risk_score(df)
        
        return df
    
    def calculate_risk_score(self, df):
        """Calculate outbreak risk score based on water quality parameters"""
        risk_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # pH risk (6.5-8.5 is safe)
            if "pH" in row and pd.notna(row["pH"]):
                if row["pH"] < 6.0 or row["pH"] > 9.0:
                    score += 3
                elif row["pH"] < 6.5 or row["pH"] > 8.5:
                    score += 2
                elif row["pH"] < 6.8 or row["pH"] > 8.2:
                    score += 1
            
            # Turbidity risk (>10 NTU is concerning)
            if "turbidity" in row and pd.notna(row["turbidity"]) and row["turbidity"] > 10:
                score += min(3, int(row["turbidity"] / 10))
            
            # TDS risk (>500 ppm is concerning)
            if "tds" in row and pd.notna(row["tds"]) and row["tds"] > 500:
                score += min(3, int(row["tds"] / 250))
            
            # Temperature risk (extreme temperatures)
            if "temperature" in row and pd.notna(row["temperature"]):
                if row["temperature"] > 35 or row["temperature"] < 10:
                    score += 2
            
            risk_scores.append(min(10, score))  # Cap at 10
        
        return risk_scores
    
    def save_prediction(self, prediction_data):
        """Save prediction results to database"""
        try:
            db = self.connect_mongo()
            if db is not None:
                collection = db["predictions"]
                prediction_data["timestamp"] = datetime.now()
                collection.insert_one(prediction_data)
                return True
            else:
                print("⚠️ MongoDB connection not available")
                return False
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions from database"""
        try:
            db = self.connect_mongo()
            if db is not None:
                collection = db["predictions"]
                predictions = list(collection.find().sort("timestamp", -1).limit(limit))
                return predictions
            else:
                print("⚠️ MongoDB connection not available")
                return []
        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return []

# Singleton instance
db_connector = DatabaseConnector()
