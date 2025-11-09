#!/bin/bash
# Quick demonstration of LNIRT model functionality

echo "======================================================================"
echo "LNIRT MODEL QUICK DEMO"
echo "======================================================================"
echo ""
echo "This demo trains a model on a small dataset and makes predictions."
echo "For full dataset training, use: python3 cli.py train --data-file data/sample_training_data.csv"
echo ""

# Create small demo dataset
echo "1. Creating demo dataset (100 samples)..."
head -101 data/sample_training_data.csv > data/demo_data.csv
echo "   ✓ Created data/demo_data.csv"
echo ""

# Train model
echo "2. Training LNIRT model (this takes 1-2 minutes)..."
python3 cli.py train --data-file data/demo_data.csv --max-iter 50 --model-file models/demo_model.pkl
echo ""

# Make predictions
echo "3. Making sample predictions..."
echo ""
echo "--- Prediction 1: Known user on known item ---"
python3 cli.py predict --user-id user_000 --item-id item_001 --model-file models/demo_model.pkl
echo ""

echo "--- Prediction 2: New user ---"
python3 cli.py predict --user-id new_student --item-id item_001 --model-file models/demo_model.pkl
echo ""

echo "--- Prediction 3: New difficult item with features ---"
python3 cli.py predict --user-id user_000 --item-id hard_task \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 5.0}' \
  --model-file models/demo_model.pkl
echo ""

# Show stats
echo "4. Model statistics..."
python3 cli.py stats --model-file models/demo_model.pkl --top-users 5 --hardest-items 5
echo ""

echo "======================================================================"
echo "DEMO COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  - Train on full dataset: python3 cli.py train --data-file data/sample_training_data.csv"
echo "  - Add your own data: Use data/user_data_template.csv as a guide"
echo "  - See README.md for detailed documentation"
echo ""
