# Topic-Based LNIRT Model - Usage Guide

## Overview

The LNIRT model now supports **topic-based training and predictions**, allowing you to:
- Train on data from multiple subjects/topics
- Make predictions filtered by topic
- Get topic-specific statistics
- Use topic-aware population averages for new items

## Available Topics

### Mathematics (5 topics)
- `algebra` - Basic algebra, equations, factoring
- `geometry` - Areas, volumes, angles, shapes
- `calculus` - Derivatives, integrals, limits
- `statistics` - Mean, median, probability
- `trigonometry` - Sin, cos, tan, angles

### Economics (3 topics)
- `microeconomics` - Supply/demand, markets, pricing
- `macroeconomics` - GDP, inflation, monetary policy
- `global_economics` - Trade, exchange rates, globalization

## Quick Start

### 1. Train on Topic-Specific Data

```bash
# Train on full topic dataset (4800 samples, 8 topics)
python3 cli.py train --data-file data/topic_training_data.csv --max-iter 100

# For quick testing (600 samples)
python3 cli.py train --data-file data/topic_test_data.csv --max-iter 50 --model-file models/topic_model.pkl
```

### 2. Make Predictions

#### For Items in Training Data

```bash
# Predict without specifying topic (works fine)
python3 cli.py predict --user-id user_000 --item-id algebra_003

# Predict with topic validation
python3 cli.py predict --user-id user_000 --item-id algebra_003 --topic algebra
```

#### For NEW Items (using sample tasks)

```bash
# Easy algebra problem (difficulty 1, time 30s)
python3 cli.py predict --user-id user_000 --item-id algebra_new_001 \
  --topic algebra \
  --item-features '{"a": 1.2, "b": -1.0, "beta": 3.4}'

# Hard calculus problem (difficulty 4, time 120s)
python3 cli.py predict --user-id user_000 --item-id calculus_hard \
  --topic calculus \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 4.8}'

# Microeconomics question (difficulty 3, time 110s)
python3 cli.py predict --user-id user_015 --item-id micro_new \
  --topic microeconomics \
  --item-features '{"a": 1.5, "b": 0.5, "beta": 4.7}'
```

### 3. View Statistics

```bash
# All topics
python3 cli.py stats

# Specific topic only
python3 cli.py stats --topic calculus --hardest-items 10
python3 cli.py stats --topic microeconomics --hardest-items 5
```

## Using Sample Tasks

The `sample_tasks/` directory contains 50 real questions you can test with.

### Example Workflow

1. **Pick a task from a file:**
```bash
cat sample_tasks/calculus_tasks.txt
```

Output:
```
calculus_007 | Find critical points of f(x) = x³ - 6x² + 9x | 4 | 120
```

2. **Convert to item features:**
- Difficulty 4 → `b = 1.5`
- Time 120s → `beta = 4.8`
- Use `a = 2.0` (good discrimination)

3. **Make prediction:**
```bash
python3 cli.py predict --user-id user_000 --item-id calculus_007 \
  --topic calculus \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 4.8}'
```

4. **Result:**
```
Probability of Correct Answer: 4.9%
Expected Response Time: 165.4 seconds (2.8 minutes)
```

## Feature: Topic-Aware Population Averages

When predicting for a **new item** with a **topic specified**, the model uses **topic-specific averages** instead of global averages:

```bash
# Without topic: uses global averages (a=1.0, b=0.0, beta=0.0)
python3 cli.py predict --user-id user_000 --item-id brand_new_item

# With topic: uses calculus-specific averages
python3 cli.py predict --user-id user_000 --item-id brand_new_item --topic calculus
```

This gives more accurate predictions for new items when you know the topic!

## Conversion Guide: Task Features → Model Parameters

### Difficulty → b (difficulty parameter)

| Difficulty Level | b value | Example |
|-----------------|---------|---------|
| 1 (very easy) | -1.0 to -0.5 | "Solve for x: 2x + 5 = 15" |
| 2 (easy) | -0.5 to 0.0 | "Factor: x² + 7x + 12" |
| 3 (medium) | 0.0 to 0.5 | "Integrate: ∫(2x + 3)dx" |
| 4 (hard) | 0.5 to 1.5 | "Find critical points of f(x) = x³ - 6x² + 9x" |
| 5 (very hard) | 1.5 to 2.5 | "Apply multivariable chain rule" |

### Time → beta (time intensity parameter)

| Time Range | beta value | exp(beta) ≈ actual time |
|------------|------------|------------------------|
| < 40 seconds | 3.0 - 3.5 | ~20-33 seconds |
| 40-70 seconds | 3.5 - 4.2 | ~33-67 seconds |
| 70-120 seconds | 4.2 - 4.8 | ~67-122 seconds |
| 120-180 seconds | 4.8 - 5.2 | ~122-181 seconds |
| > 180 seconds | 5.2 - 5.8 | ~181-330 seconds |

### Discrimination → a parameter

- Use `a = 1.0 - 1.5` for most tasks
- Use `a = 1.5 - 2.5` for tasks that strongly test ability

## Example Test Scenarios

### Scenario 1: Compare Same User Across Topics

```bash
# User on easy algebra
python3 cli.py predict --user-id user_000 --item-id algebra_001 --topic algebra

# Same user on hard calculus
python3 cli.py predict --user-id user_000 --item-id calculus_007 \
  --topic calculus \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 4.8}'

# Same user on microeconomics
python3 cli.py predict --user-id user_000 --item-id microeconomics_001 \
  --topic microeconomics \
  --item-features '{"a": 1.5, "b": 0.2, "beta": 4.3}'
```

### Scenario 2: Compare Different Users on Same Task

```bash
# Low ability user
python3 cli.py predict --user-id user_001 --item-id algebra_005 --topic algebra

# Average user
python3 cli.py predict --user-id user_030 --item-id algebra_005 --topic algebra

# High ability user
python3 cli.py predict --user-id user_007 --item-id algebra_005 --topic algebra
```

### Scenario 3: Explore Topic Difficulty

```bash
# View hardest problems in each topic
python3 cli.py stats --topic algebra --hardest-items 5
python3 cli.py stats --topic calculus --hardest-items 5
python3 cli.py stats --topic microeconomics --hardest-items 5
```

## Data Format for Training

Your training CSV must include a `topic` column:

```csv
user_id,item_id,topic,correct,response_time
user_001,algebra_01,algebra,1,45.2
user_001,calculus_05,calculus,0,180.5
user_002,micro_03,microeconomics,1,95.3
```

## How Topics Work Internally

1. **During Training:**
   - Model extracts topic information from data
   - Stores topic with each item's parameters
   - Calculates topic-specific averages

2. **During Prediction (with --topic):**
   - Validates item belongs to specified topic (if item exists)
   - For new items: uses topic-specific population averages
   - Makes prediction using appropriate parameters

3. **Statistics:**
   - `--topic` flag filters items by topic
   - Shows topic-specific difficulty distributions

## Testing Your Understanding

Try these commands to verify the system works:

```bash
# 1. Train the model
python3 cli.py train --data-file data/topic_test_data.csv --max-iter 30 --model-file models/my_test.pkl

# 2. List available topics
python3 cli.py stats --model-file models/my_test.pkl | grep "Topics"

# 3. Predict on a training item
python3 cli.py predict --user-id user_000 --item-id algebra_003 --model-file models/my_test.pkl

# 4. Predict on a new calculus problem
python3 cli.py predict --user-id user_000 --item-id new_calc --topic calculus \
  --item-features '{"a": 1.8, "b": 0.8, "beta": 4.5}' \
  --model-file models/my_test.pkl

# 5. View calculus-specific stats
python3 cli.py stats --topic calculus --model-file models/my_test.pkl
```

## Files Reference

| File | Description |
|------|-------------|
| `data/topic_training_data.csv` | 4800 training samples, 8 topics |
| `data/topic_test_data.csv` | 600 test samples, 8 topics |
| `data/topic_item_bank.json` | 180 items with metadata |
| `sample_tasks/*.txt` | 50 real questions for testing |
| `sample_tasks/README.md` | Detailed usage guide for sample tasks |
| `models/topic_model.pkl` | Pre-trained model on test data |

## Tips

1. **Always specify topic for new items** - gives better predictions via topic-specific averages
2. **Use sample task files** - they have pre-written questions with difficulty/time estimates
3. **Check topic statistics first** - understand topic difficulty before making predictions
4. **Start with trained items** - test existing items before creating new ones

## Troubleshooting

**Q: "Item belongs to topic 'unknown', not 'calculus'"**

A: The item exists in training data but topic wasn't loaded. Retrain the model with verbose flag to see if topics are detected.

**Q: My predictions seem wrong**

A: Check if you're using the right parameter ranges. Use the conversion tables above.

**Q: How do I know what users are available?**

A: Run `python3 cli.py stats --top-users 60` to see all users and their abilities.

## Next Steps

Ready to integrate with SmartStudy? See the main README.md for integration guidelines.
