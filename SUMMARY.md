# SmartStudy ML System - Summary

## What This Is

Complete ML prediction system for IB Diploma Programme student performance using topic-based LNIRT models.

## Key Innovation

**Complete topic separation** - Each subject (calculus, algebra, economics, etc.) has its own independent model, ensuring accurate topic-specific predictions.

## What It Does

1. **Predicts correctness** - Will the student answer correctly?
2. **Predicts time** - How long will it take?
3. **Learns continuously** - Gets more accurate as students complete tasks
4. **Personalizes** - Adapts to each individual student's ability and speed

## Architecture

```
Input: user_id + topic + difficulty (1-3)
  ↓
Topic-Specific Model (e.g., calculus)
  ↓
Output: probability_correct + expected_time
  ↓
Save to Database
  ↓
After completion: Update with actual results
  ↓
Dual Learning:
  1. Personalized (immediate, per-user)
  2. General (periodic, all users)
```

## Files

| File | Purpose |
|------|---------|
| `smart_cli.py` | CLI interface (train/predict/update/stats) |
| `topic_lnirt.py` | LNIRT model implementation |
| `predictions_db.py` | SQLite database management |
| `generate_ib_data.py` | IB DP data generator |
| `SMARTSTUDY_README.md` | Full documentation |
| `data/ib/*.csv` | Training data (10 topics) |
| `models/*.pkl` | Trained models |
| `predictions.db` | Predictions database |

## Quick Start

```bash
# Train
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv

# Predict
python3 smart_cli.py predict --user-id student_001 --topic calculus --difficulty 2 --save

# Update
python3 smart_cli.py update --task-id 1 --correct 1 --time 180

# Stats
python3 smart_cli.py stats --topic calculus
```

## Available Topics

**Math AA (7):** numbers, algebra, functions, geometry, trigonometry, calculus, statistics

**Economics (3):** microeconomics, macroeconomics, global_economics

## Training Data

- **1500 samples per topic** (50 users × 30 responses)
- **3 difficulty levels** per topic
- **Realistic performance patterns** based on IB DP standards

## How It Works

### Topic Separation
- Calculus model ≠ Algebra model ≠ Economics model
- Each trained only on its topic's data
- Predictions are topic-specific

### Personalization
- **First prediction:** Uses general parameters
- **After updates:** Uses personalized user parameters
- **Result:** Predictions improve over time for each user

### Dual Learning
1. **Personalized:** Updates immediately after each task
2. **General:** Retrains periodically (every 50 responses)

## Example

```bash
# New user, first prediction on calculus
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
→ 13% correct, 332s (using general parameters)

# Alice completes the task
$ python3 smart_cli.py update --task-id 1 --correct 1 --time 180
→ Model learns!

# Second prediction for alice on calculus
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2
→ 17% correct, 277s (personalized!)

# Same user, different topic
$ python3 smart_cli.py predict --user-id alice --topic algebra --difficulty 2
→ 85% correct, 75s (different model!)
```

## Testing Results

✅ Topic separation verified (different predictions per topic)
✅ Personalization verified (predictions improve over time)
✅ Database tracking working
✅ Error handling robust
✅ All commands functional

## Integration

```python
from topic_lnirt import TopicModelManager
from predictions_db import PredictionsDB

manager = TopicModelManager()
db = PredictionsDB()

# Predict
model = manager.get_model('calculus')
p_correct, time = model.predict('student_001', difficulty=2)

# Save
task_id = db.add_prediction('student_001', 'calculus', 2, p_correct, time)

# Update (after completion)
db.update_prediction(task_id, actual_correct=1, actual_time=180)
model.update_from_response('student_001', 2, 1, 180)
manager.save_model('calculus')
```

## Documentation

See **SMARTSTUDY_README.md** for complete documentation including:
- Detailed CLI reference
- API integration examples
- Troubleshooting guide
- Technical specifications

## GitHub

**Repository:** https://github.com/DDDime77/mlstudy
