# CLI Testing Guide

Quick reference for testing the SmartStudy LNIRT prediction system.

---

## Prerequisites

System is already set up with:
- ✓ 50 users trained (user_000 through user_049)
- ✓ 1500 completed tasks in database
- ✓ Trained calculus model at `models/calculus.pkl`

---

## Quick Start: Test the Complete Workflow

### 1. Make a Prediction (New User)

```bash
python3 smart_cli.py predict --user-id testuser --topic calculus --difficulty 2 --save
```

**Expected output:**
```
Probability of Correct: 32.4%
Expected Time: 121.8 seconds
✓ Prediction saved with task_id=XXXX
```

**Note the task_id** - you'll need it for the next step.

### 2. Update with Actual Results

Replace `XXXX` with your actual task_id:

```bash
python3 smart_cli.py update --task-id XXXX --correct 1 --time 95
```

**Expected output:**
```
✓ Database updated

AUTOMATIC USER-SPECIFIC TRAINING
Training personalized model for testuser...
  Using 1 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.676

  ✓ User parameters updated
    Ability (θ): 2.XXX
    Speed (τ): 0.XXX

✓ Personalized model updated for testuser
```

### 3. Make Second Prediction (Now Personalized!)

```bash
python3 smart_cli.py predict --user-id testuser --topic calculus --difficulty 2 --save
```

**Expected output:**
```
Probability of Correct: 75-80%  (was 32.4%)
Expected Time: 75-85 seconds    (was 121.8s)
```

**✓ Success!** The prediction improved dramatically because the system learned from your first task.

---

## Core Commands

### PREDICT - Make a Prediction

```bash
python3 smart_cli.py predict --user-id <USER> --topic <TOPIC> --difficulty <1|2|3> [--save]
```

**Examples:**

```bash
# New user (will use population average)
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save

# Existing user (will use personalized parameters)
python3 smart_cli.py predict --user-id user_010 --topic calculus --difficulty 2 --save

# Different difficulties
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 1  # Easy
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 3  # Hard

# Without --save (just see prediction, don't save)
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2
```

### UPDATE - Record Actual Results (AUTO-TRAINS!)

```bash
python3 smart_cli.py update --task-id <ID> --correct <0|1> --time <SECONDS>
```

**Examples:**

```bash
# Correct answer, took 95 seconds
python3 smart_cli.py update --task-id 1501 --correct 1 --time 95

# Incorrect answer, took 180 seconds
python3 smart_cli.py update --task-id 1502 --correct 0 --time 180

# With custom retrain threshold
python3 smart_cli.py update --task-id 1503 --correct 1 --time 88 --retrain-threshold 100
```

**What happens automatically:**
1. Database updated with actual results
2. User-specific training triggered immediately
3. Personalized parameters saved
4. Next prediction will use new parameters

### STATS - View Statistics

```bash
python3 smart_cli.py stats [--topic <TOPIC>]
```

**Examples:**

```bash
# List all topics
python3 smart_cli.py stats

# Detailed stats for calculus
python3 smart_cli.py stats --topic calculus
```

**Output includes:**
- Difficulty parameters for each level
- User ability/speed distributions
- Prediction history
- Actual performance by difficulty

### TRAIN - Train Model

```bash
python3 smart_cli.py train --topic <TOPIC> [OPTIONS]
```

**Examples:**

```bash
# General training (retrain on new data)
python3 smart_cli.py train --topic calculus --stats

# Manual user-specific training (rarely needed, auto happens on update)
python3 smart_cli.py train --topic calculus --user-id alice

# Initial training from CSV
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv
```

---

## Testing Scenarios

### Scenario A: Test New User Journey

```bash
# Step 1: First prediction (uses population average)
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
# Note: P(correct) ≈ 32%, Time ≈ 122s
# Remember task_id

# Step 2: Update with good performance
python3 smart_cli.py update --task-id XXXX --correct 1 --time 85
# Note: Automatic training happens, θ increases

# Step 3: Second prediction (now personalized)
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
# Note: P(correct) ≈ 75-80%, Time ≈ 75-85s (BIG improvement!)

# Step 4: Update again
python3 smart_cli.py update --task-id YYYY --correct 1 --time 82
# Note: Parameters refined

# Step 5: Third prediction (more accurate)
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2
# Note: Predictions stabilizing
```

### Scenario B: Test Existing User

```bash
# Use one of the 50 CSV users
python3 smart_cli.py predict --user-id user_010 --topic calculus --difficulty 2
# Note: Uses parameters from general training (not population average)
```

### Scenario C: Test Different Difficulties

```bash
# Easy problem
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 1
# Note: Higher P(correct), lower time

# Medium problem
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2
# Note: Moderate P(correct) and time

# Hard problem
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 3
# Note: Lower P(correct), higher time
```

### Scenario D: Test Poor Performance

```bash
# Predict
python3 smart_cli.py predict --user-id bob --topic calculus --difficulty 2 --save
# Note task_id

# Fail the task
python3 smart_cli.py update --task-id XXXX --correct 0 --time 200
# Note: θ will DECREASE (appropriate!)

# Next prediction
python3 smart_cli.py predict --user-id bob --topic calculus --difficulty 2
# Note: Lower P(correct) than if they had succeeded
```

---

## Quick Verification Commands

### Check if User is Trained

```bash
python3 -c "
from topic_lnirt import TopicModelManager
model = TopicModelManager().get_model('calculus')
user = 'alice'
if user in model.user_params:
    print(f'{user} is trained: θ={model.user_params[user][\"theta\"]:.3f}, τ={model.user_params[user][\"tau\"]:.3f}')
else:
    print(f'{user} is NOT trained (will use population average)')
"
```

### Check User's History

```bash
python3 -c "
import pandas as pd
from predictions_db import PredictionsDB
db = PredictionsDB()
history = pd.read_sql_query(
    \"SELECT COUNT(*) as count, AVG(actual_correct) as accuracy, AVG(actual_time) as avg_time
     FROM predictions
     WHERE user_id='alice' AND topic='calculus'\",
    db.conn
).iloc[0]
print(f'Tasks completed: {history[\"count\"]:.0f}')
if history['count'] > 0:
    print(f'Accuracy: {history[\"accuracy\"]:.1%}')
    print(f'Avg time: {history[\"avg_time\"]:.1f}s')
db.close()
"
```

### View Latest Prediction

```bash
python3 -c "
import pandas as pd
from predictions_db import PredictionsDB
db = PredictionsDB()
latest = pd.read_sql_query(
    \"SELECT task_id, user_id, difficulty, predicted_correct, predicted_time,
            actual_correct, actual_time, created_at
     FROM predictions
     WHERE topic='calculus'
     ORDER BY task_id DESC
     LIMIT 5\",
    db.conn
)
print(latest.to_string(index=False))
db.close()
"
```

### Check Population Average

```bash
python3 -c "
import numpy as np
from topic_lnirt import TopicModelManager
model = TopicModelManager().get_model('calculus')
avg_theta = np.mean([p['theta'] for p in model.user_params.values()])
avg_tau = np.mean([p['tau'] for p in model.user_params.values()])
print(f'Population average: θ={avg_theta:.3f}, τ={avg_tau:.3f}')
print(f'This is what new users get')
"
```

---

## Testing the Tests

### Run Comprehensive Tests

```bash
# All 31 comprehensive tests
python3 comprehensive_error_analysis.py
# Expected: 31/31 PASS

# Automatic training workflow tests
python3 test_automatic_training.py
# Expected: 11/11 PASS

# Three scenarios comparison
python3 test_three_scenarios.py
# Shows comparison of new user, CSV user, and trained user
```

---

## Common Patterns You'll See

### Pattern 1: New User
```
First prediction: 32.4% / 121.8s (population average)
                      ↓
After 1 task:     70-80% / 75-90s (personalized, BIG jump)
                      ↓
After 3 tasks:    Stabilizes around true ability
```

### Pattern 2: Existing User (from CSV)
```
First prediction: 30-50% / 110-140s (their historical performance)
                      ↓
After update:     Similar (already well-calibrated from 30 tasks)
```

### Pattern 3: Difficulty Effect
```
Difficulty 1:  50-70% correct, 60-90s
Difficulty 2:  30-50% correct, 90-130s
Difficulty 3:  20-40% correct, 120-180s
```

### Pattern 4: Performance Impact
```
Success (correct=1):    θ increases → higher future P(correct)
Failure (correct=0):    θ decreases → lower future P(correct)
Fast (time < expected): τ increases → lower future expected time
Slow (time > expected): τ decreases → higher future expected time
```

---

## Troubleshooting

### "Model for topic 'calculus' is not trained"
```bash
# Train the model first
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv
```

### "Task ID not found"
Check you're using the correct task_id from the prediction output.

### No automatic training happening
Make sure you used `--save` when making the prediction.

### Predictions seem wrong
```bash
# Check model stats
python3 smart_cli.py stats --topic calculus

# Check user history
python3 -c "from predictions_db import PredictionsDB; import pandas as pd; db = PredictionsDB(); print(pd.read_sql_query(\"SELECT * FROM predictions WHERE user_id='youruser'\", db.conn)); db.close()"
```

---

## Quick Reference Table

| Command | Purpose | Auto-Trains? |
|---------|---------|--------------|
| `predict --save` | Make prediction, save to DB | No |
| `update` | Record actual results | **YES** |
| `train` | General training | Only if --user-id |
| `stats` | View statistics | No |

---

## Typical Usage Session

```bash
# 1. Start fresh with new user
python3 smart_cli.py predict --user-id student_x --topic calculus --difficulty 2 --save
# Output: task_id=1234, P=32.4%, T=121.8s

# 2. Student completes task (correct, 95s)
python3 smart_cli.py update --task-id 1234 --correct 1 --time 95
# Output: Automatic training, θ=2.8, τ=0.4

# 3. Next prediction
python3 smart_cli.py predict --user-id student_x --topic calculus --difficulty 2 --save
# Output: task_id=1235, P=78.5%, T=76.2s

# 4. Student completes task (correct, 82s)
python3 smart_cli.py update --task-id 1235 --correct 1 --time 82
# Output: Automatic training, θ=2.9, τ=0.35

# 5. Check stats
python3 smart_cli.py stats --topic calculus

# 6. Repeat as needed...
```

---

## Notes

- **All task_ids are unique** - each prediction gets a new one
- **Auto-training is immediate** - happens right after update
- **Parameters are bounded** - θ and τ stay between -3 and +3
- **Predictions are deterministic** - same input = same output
- **Each topic is independent** - different models for different subjects

---

## Testing Checklist

Use this to verify everything works:

- [ ] Make prediction for new user (should use pop avg: ~32% / ~122s)
- [ ] Save prediction and get task_id
- [ ] Update with actual results
- [ ] See automatic training output
- [ ] Make second prediction (should be personalized: 70-80%)
- [ ] Update again and see parameter refinement
- [ ] Make prediction for existing user (user_010)
- [ ] Try all three difficulty levels
- [ ] View stats for calculus
- [ ] Run comprehensive_error_analysis.py (31/31 pass)
- [ ] Run test_automatic_training.py (11/11 pass)

---

That's it! The system is ready to use. Start with the Quick Start section and explore from there.
