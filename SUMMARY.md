# LNIRT Model Playground - Project Summary

## What Was Built

A complete, working implementation of the **LNIRT (Joint IRT + Lognormal Response Time) Model** for predicting:
1. **Task completion probability** - Will the user answer correctly?
2. **Task completion time** - How long will it take?

## Project Structure

```
ml_lnirt_playground/
├── cli.py                      # Main CLI interface ⭐
├── lnirt_model.py             # Core LNIRT model implementation
├── generate_sample_data.py    # Data generation utilities
├── quick_demo.sh              # Quick demonstration script
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── SUMMARY.md                 # This file
├── data/
│   ├── sample_training_data.csv    # 1500 training samples
│   ├── item_bank.json              # 100 question items
│   ├── user_data_template.csv      # Template for your data
│   ├── demo_data.csv              # Small demo dataset
│   └── test_data.csv              # Test dataset
└── models/
    ├── demo_model.pkl         # Pre-trained demo model
    └── test_model.pkl         # Test model
```

## Quick Start

### Option 1: Try Predictions Immediately (Using Pre-trained Model)

```bash
# Predict for a user on a task
python3 cli.py predict --user-id user_000 --item-id item_001 --model-file models/demo_model.pkl

# Predict for a new difficult task
python3 cli.py predict --user-id user_000 --item-id hard_question \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 5.0}' \
  --model-file models/demo_model.pkl

# View model statistics
python3 cli.py stats --model-file models/demo_model.pkl
```

### Option 2: Run Full Demo

```bash
./quick_demo.sh
```

### Option 3: Train Your Own Model

```bash
# Train on the full sample dataset (1500 responses)
# Note: This takes 5-10 minutes with max_iter=200+
python3 cli.py train --data-file data/sample_training_data.csv --max-iter 100

# For faster training (less optimal):
python3 cli.py train --data-file data/sample_training_data.csv --max-iter 50

# With statistics output:
python3 cli.py train --data-file data/sample_training_data.csv --max-iter 100 --stats
```

## CLI Commands Reference

### 1. Train Model

```bash
python3 cli.py train --data-file <path> [options]

Options:
  --data-file, -d     Path to training CSV (required)
  --model-file, -m    Where to save model (default: models/lnirt_model.pkl)
  --max-iter          Optimization iterations (default: 1000)
  --stats             Show statistics after training
  --verbose, -v       Detailed training output
```

### 2. Make Predictions

```bash
python3 cli.py predict --user-id <id> --item-id <id> [options]

Options:
  --user-id, -u       User identifier (required)
  --item-id, -i       Item/task identifier (required)
  --item-features, -f JSON with item parameters (for new items)
  --model-file, -m    Path to trained model
```

### 3. View Statistics

```bash
python3 cli.py stats [options]

Options:
  --model-file, -m    Path to trained model
  --top-users         Show N top users by ability
  --hardest-items     Show N hardest items
```

## Sample Data Explained

### 1. `data/sample_training_data.csv` (1500 samples)
**Purpose:** Pre-generated training data simulating realistic educational assessment data

**Statistics:**
- 50 unique users (students)
- 100 unique items (questions)
- 30 responses per user
- Overall accuracy: ~45%
- Mean response time: ~96 seconds

**Why it's included:** Allows immediate testing without real data collection. Generated using realistic IRT parameters.

### 2. `data/item_bank.json`
**Purpose:** Question metadata and classification

**Contains:**
- 100 items with IDs matching training data
- Sample question text
- Topic tags (algebra, geometry, calculus, statistics, trigonometry)
- Difficulty ratings (1-5)

**Use case:** Demonstrates how to organize question metadata for intelligent task selection.

### 3. `data/user_data_template.csv`
**Purpose:** Template for adding your own training data

**Format:**
```csv
user_id,item_id,correct,response_time,notes
user_001,item_001,1,45.2,"Optional notes"
```

**Required columns:**
- `user_id`: String identifier
- `item_id`: String identifier
- `correct`: 1 (correct) or 0 (incorrect)
- `response_time`: Seconds (can be decimal)

## What the Model Learns

### User Parameters (Learned for Each User)
- **θ (theta) - Ability**: How capable the user is
  - Range: typically -3 to +3
  - Higher = more capable

- **τ (tau) - Speed**: How fast the user works
  - Range: typically -2 to +2
  - Higher = faster

### Item Parameters (Learned for Each Task)
- **a - Discrimination**: How well the item distinguishes ability
  - Range: 0.1 to 5
  - Higher = better at distinguishing

- **b - Difficulty**: How hard the item is
  - Range: typically -3 to +3
  - Higher = harder

- **β (beta) - Time Intensity**: How long the item takes
  - Range: 3 to 6 (in log-seconds)
  - exp(3) ≈ 20s, exp(5) ≈ 150s, exp(6) ≈ 400s

## Integration with SmartStudy

### Use Case 1: Show Expected Time to Users
```python
p_correct, expected_time = model.predict(user_id, task_id)
print(f"This task will take approximately {expected_time/60:.0f} minutes")
```

### Use Case 2: Adaptive Task Selection
```python
# After user completes a task
predicted_prob, _ = model.predict(user_id, completed_task_id)
actual_result = user_answered_correctly

if actual_result and predicted_prob < 0.5:
    # User exceeded expectations → increase difficulty
    next_task = select_harder_task(user_id)

elif not actual_result and predicted_prob > 0.5:
    # User underperformed → decrease difficulty
    next_task = select_easier_task(user_id)
```

### Use Case 3: Progress Tracking
```python
user_stats = model.get_user_stats()
user_ability = user_stats[user_stats['user_id'] == current_user]['ability_theta'].values[0]

# Track ability over time as user learns
# Retrain model periodically with new data
```

## Training Performance Notes

**Training time depends on:**
- Number of users × 2 parameters
- Number of items × 3 parameters
- Number of responses
- `max_iter` setting

**Example timings:**
- 30 responses (1 user, 30 items): ~15-20 seconds (50 iterations)
- 100 responses (few users): ~45-60 seconds (50 iterations)
- 1500 responses (50 users, 100 items): ~5-10 minutes (100 iterations)

**Recommendations:**
- For development/testing: `--max-iter 50` (faster, less optimal)
- For production: `--max-iter 200-500` (slower, more accurate)
- With good initialization, 100 iterations often sufficient

## Tested Functionality ✓

All features have been tested and verified:

- ✅ Training on provided data
- ✅ Saving and loading models
- ✅ Predictions for known users/items
- ✅ Predictions for new users (uses population average)
- ✅ Predictions for new items with features
- ✅ Model statistics display
- ✅ Data format validation
- ✅ Error handling and helpful messages

## Next Steps

### For Immediate Use:
1. Try the pre-trained model: `python3 cli.py predict --user-id user_000 --item-id item_001 --model-file models/demo_model.pkl`
2. Run the demo: `./quick_demo.sh`
3. Read the full README.md for detailed documentation

### For SmartStudy Integration:
1. Set up data collection in smartstudy to capture:
   - user_id
   - task_id
   - correctness (0/1)
   - response_time (seconds)

2. Periodically export this data to CSV format

3. Train the model on your real data:
   ```bash
   python3 cli.py train --data-file smartstudy_data.csv --max-iter 200
   ```

4. Use the model for predictions in your application:
   ```python
   from lnirt_model import LNIRTModel

   model = LNIRTModel()
   model.load_model('models/lnirt_model.pkl')

   p_correct, expected_time = model.predict(user_id, task_id)
   ```

5. Retrain periodically (e.g., nightly) as more data is collected

## Technical Details

**Model Type:** Joint LNIRT (IRT + Lognormal Response Time)
**Optimization:** L-BFGS-B (maximum likelihood estimation)
**Libraries:** NumPy, SciPy, Pandas
**Python Version:** 3.7+

## Support

For detailed documentation, see:
- `README.md` - Complete user guide
- `lnirt_model.py` - Implementation with inline documentation
- `cli.py` - CLI interface with examples

## License

This is a playground/prototype implementation for the smartstudy project.
Modify and use as needed.
