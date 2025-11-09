# SmartStudy LNIRT System

**Complete topic-based prediction system for IB Diploma Programme subjects**

## Overview

This system provides **accurate predictions** of student performance using Item Response Theory (IRT) combined with response time modeling. Each topic is completely independent, ensuring accurate topic-specific predictions.

### Key Features

- ✅ **Topic Separation**: Each subject (calculus, algebra, economics, etc.) has its own independent model
- ✅ **Simple Interface**: Predict using only `topic` + `difficulty` (1-3) + `user_id`
- ✅ **Dual Learning**: Both general (all users) and personalized (per user) learning
- ✅ **Database Tracking**: All predictions and actual results are saved
- ✅ **IB DP Aligned**: Pre-generated data for Math AA and Economics subjects

## Quick Start

### 1. Train Models

Train models for the topics you need:

```bash
# Math AA topics
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv
python3 smart_cli.py train --topic algebra --data-file data/ib/algebra.csv
python3 smart_cli.py train --topic geometry --data-file data/ib/geometry.csv

# Economics topics
python3 smart_cli.py train --topic microeconomics --data-file data/ib/microeconomics.csv
python3 smart_cli.py train --topic macroeconomics --data-file data/ib/macroeconomics.csv
```

### 2. Make Predictions

Predict for a user on a specific topic and difficulty:

```bash
python3 smart_cli.py predict \
  --user-id student_042 \
  --topic calculus \
  --difficulty 2 \
  --save
```

**Output:**
```
Probability of Correct: 15.3%
Expected Time: 285.2 seconds (4.8 minutes)
✗ Confidence: LOW (15%)
⏳ Significant time required (4.8 min)

✓ Prediction saved with task_id=5
```

### 3. Update with Actual Results

After the student completes the task:

```bash
python3 smart_cli.py update \
  --task-id 5 \
  --correct 1 \
  --time 240
```

This automatically:
- Updates the database
- Applies **personalized learning** for this user
- Accumulates data for **general model retraining**

## Available Topics

### Math AA (7 topics)
- `numbers` - Number systems, arithmetic
- `algebra` - Equations, systems, polynomials
- `functions` - Function analysis, transformations
- `geometry` - Shapes, areas, volumes, 3D
- `trigonometry` - Trig functions, identities, unit circle
- `calculus` - Derivatives, integrals, limits
- `statistics` - Probability, distributions, analysis

### Economics (3 topics)
- `microeconomics` - Markets, supply/demand, pricing
- `macroeconomics` - GDP, inflation, monetary policy
- `global_economics` - Trade, exchange rates, globalization

## Difficulty Levels

| Level | Description | Typical Accuracy | Typical Time |
|-------|-------------|------------------|--------------|
| 1 | Easy | 60-80% | 30-60s |
| 2 | Medium | 40-60% | 60-120s |
| 3 | Hard | 20-45% | 120-300s |

*Values vary by topic*

## How It Works

### Topic Separation

**Each topic is COMPLETELY INDEPENDENT:**

- Calculus model: trained only on calculus data
- Algebra model: trained only on algebra data
- etc.

This ensures predictions are accurate for each subject.

```bash
# These use DIFFERENT models
python3 smart_cli.py predict --user-id user_001 --topic calculus --difficulty 2
# → 15% correct, 285s

python3 smart_cli.py predict --user-id user_001 --topic algebra --difficulty 2
# → 85% correct, 75s
```

### Dual Learning System

#### 1. General Learning (All Users)
- Uses ALL users' data for a topic
- Updates difficulty parameters (how hard level 1/2/3 are)
- Runs periodically (every 50 responses by default)

#### 2. Personalized Learning (Per User)
- Tracks each user's performance
- Updates user-specific ability and speed
- Happens IMMEDIATELY after each update
- Makes predictions more accurate over time

**Example:**
```
First prediction for user_001 on calculus level 2:
→ 13% correct (using general parameters)

After user_001 completes task (correct in 180s):
→ Model updates user_001's parameters

Next prediction for user_001 on calculus level 2:
→ 17% correct (personalized!)
```

### Predictions Database

All predictions are tracked in `predictions.db`:

**Schema:**
- `task_id`: Unique task identifier
- `user_id`: Student identifier
- `topic`: Subject (calculus, algebra, etc.)
- `difficulty`: Level (1, 2, or 3)
- `predicted_correct`: Model's correctness prediction
- `predicted_time`: Model's time prediction
- `actual_correct`: Actual result (NULL until updated)
- `actual_time`: Actual time (NULL until updated)
- `created_at`: When prediction was made
- `updated_at`: When actual results were added

## CLI Commands

### `train` - Train Topic Model

```bash
python3 smart_cli.py train \
  --topic <topic_name> \
  --data-file <path_to_csv> \
  [--stats]
```

**Options:**
- `--topic`: Topic name (e.g., calculus, algebra)
- `--data-file`: CSV with columns: `user_id`, `difficulty`, `correct`, `response_time`
- `--stats`: Show model statistics after training

**Example:**
```bash
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv --stats
```

### `predict` - Make Prediction

```bash
python3 smart_cli.py predict \
  --user-id <user_id> \
  --topic <topic> \
  --difficulty <1|2|3> \
  [--save]
```

**Options:**
- `--user-id`: Student identifier
- `--topic`: Topic name
- `--difficulty`: Difficulty level (1=easy, 2=medium, 3=hard)
- `--save`: Save prediction to database (required for later update)

**Example:**
```bash
python3 smart_cli.py predict --user-id student_042 --topic calculus --difficulty 2 --save
```

### `update` - Update with Actual Results

```bash
python3 smart_cli.py update \
  --task-id <task_id> \
  --correct <0|1> \
  --time <seconds> \
  [--retrain-threshold <N>]
```

**Options:**
- `--task-id`: Task ID from prediction
- `--correct`: 1 if correct, 0 if incorrect
- `--time`: Actual time in seconds
- `--retrain-threshold`: Retrain model after N responses (default: 50)

**Example:**
```bash
python3 smart_cli.py update --task-id 5 --correct 1 --time 240
```

### `stats` - View Statistics

```bash
python3 smart_cli.py stats [--topic <topic>]
```

**Examples:**
```bash
# List all trained topics
python3 smart_cli.py stats

# Detailed stats for calculus
python3 smart_cli.py stats --topic calculus
```

## Complete Workflow Example

```bash
# 1. Train the model (one-time setup)
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv

# 2. Student starts a task - make prediction
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
# Output: task_id=1, predicted 15% correct, 285s

# 3. Student completes task - update with actual results
python3 smart_cli.py update --task-id 1 --correct 1 --time 240
# Model learns from this!

# 4. Next prediction for same student is personalized
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
# Output: task_id=2, predicted 19% correct, 250s (improved!)

# 5. View statistics
python3 smart_cli.py stats --topic calculus
```

## Integration with SmartStudy

### Backend Integration

```python
from topic_lnirt import TopicModelManager
from predictions_db import PredictionsDB

# Initialize
manager = TopicModelManager()
db = PredictionsDB()

# When user starts a task
model = manager.get_model(topic)
p_correct, expected_time = model.predict(user_id, difficulty)

# Save prediction
task_id = db.add_prediction(user_id, topic, difficulty, p_correct, expected_time)

# Show expected time to user
print(f"This task will take approximately {expected_time/60:.0f} minutes")

# When user completes task
db.update_prediction(task_id, actual_correct, actual_time)

# Update model (personalized learning)
model.update_from_response(user_id, difficulty, actual_correct, actual_time)
manager.save_model(topic)

# Adaptive task selection
if actual_correct and p_correct < 0.5:
    # User exceeded expectations → give harder tasks
    next_difficulty = min(difficulty + 1, 3)
elif not actual_correct and p_correct > 0.5:
    # User underperformed → give easier tasks
    next_difficulty = max(difficulty - 1, 1)
```

## File Structure

```
ml_lnirt_playground/
├── smart_cli.py              # Main CLI interface
├── topic_lnirt.py            # Topic-based LNIRT model
├── predictions_db.py         # Database management
├── generate_ib_data.py       # IB DP data generator
├── models/
│   ├── algebra.pkl           # Trained algebra model
│   ├── calculus.pkl          # Trained calculus model
│   ├── microeconomics.pkl    # Trained economics model
│   └── ...                   # Other topic models
├── data/
│   └── ib/
│       ├── algebra.csv       # Algebra training data (1500 samples)
│       ├── calculus.csv      # Calculus training data
│       ├── microeconomics.csv
│       └── ...               # All 10 topics
└── predictions.db            # SQLite database with predictions
```

## Training Data Format

CSV files must have these columns:

```csv
user_id,difficulty,correct,response_time
user_001,1,1,45.2
user_001,2,0,120.5
user_002,1,1,38.7
```

**Columns:**
- `user_id`: Student identifier (string)
- `difficulty`: 1, 2, or 3
- `correct`: 1 (correct) or 0 (incorrect)
- `response_time`: Time in seconds (float)

## Generated Data Statistics

Each IB topic includes:
- **50 users** with varying abilities
- **3 difficulty levels** (1=easy, 2=medium, 3=hard)
- **10 responses per user per difficulty**
- **Total: 1500 responses per topic**

### Topic Characteristics

| Topic | Easy Accuracy | Medium Accuracy | Hard Accuracy | Mean Time |
|-------|---------------|-----------------|---------------|-----------|
| Numbers | 87.8% | 60.6% | 46.8% | 47s |
| Algebra | 84.4% | 61.8% | 44.6% | 63s |
| Calculus | 55.8% | 33.4% | 26.4% | 126s |
| Microeconomics | 76.4% | 51.6% | 42.4% | 105s |

## Troubleshooting

### "Model for topic 'X' is not trained"

**Solution:** Train the model first:
```bash
python3 smart_cli.py train --topic X --data-file data/ib/X.csv
```

### "Task ID not found"

**Solution:** Make sure you used `--save` when making the prediction

### Predictions seem inaccurate

**Solution:**
1. Check if you have enough training data (at least 500+ responses recommended)
2. Ensure training data quality (realistic correctness/time values)
3. Use `update` command regularly to improve personalization

### Want to retrain a topic

**Solution:** Just run train again - it will overwrite the old model:
```bash
python3 smart_cli.py train --topic calculus --data-file new_data.csv
```

## Technical Details

### Model Parameters

**Per User (per topic):**
- θ (theta): Ability parameter (-3 to +3, higher = more capable)
- τ (tau): Speed parameter (-2 to +2, higher = faster)

**Per Difficulty Level:**
- a: Discrimination (0.5 to 2.5, how well it tests ability)
- b: Difficulty (-2 to +2, higher = harder)
- β (beta): Time intensity (log-seconds, higher = takes longer)

### Prediction Formulas

**Correctness:**
```
P(correct) = 1 / (1 + exp(-a × (θ - b)))
```

**Time:**
```
Expected_time = exp(β - τ)
```

## Performance

- **Training:** ~0.5-2 seconds per topic (1500 samples)
- **Prediction:** < 0.01 seconds
- **Update:** < 0.1 seconds
- **Database queries:** < 0.01 seconds

## License

Part of the SmartStudy project.
