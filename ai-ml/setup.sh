#!/bin/bash
echo "Setting up AI/ML environment for AquaGuard..."
echo "=============================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r ../ai-ml/requirements.txt

# Create models directory
echo "Creating models directory..."
mkdir -p ../models

# Create initial training data
echo "Generating initial training data..."
python generate_training_data.py

# Train initial model
echo "Training initial model..."
python train_model.py

echo "=============================================="
echo "AI/ML setup completed!"
echo "You can now start the server and use the AI features."
echo "API endpoints available:"
echo "  POST /api/ai/predict"
echo "  POST /api/ai/analyze-risk" 
echo "  GET  /api/ai/model-info"
echo "  POST /api/ai/retrain"
echo "  GET  /api/ai/predictions"
