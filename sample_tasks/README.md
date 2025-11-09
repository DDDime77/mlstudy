# Sample Tasks for Testing

This directory contains real questions organized by topic that you can use to test the LNIRT model.

## Available Topics

### Mathematics
- **algebra_tasks.txt** - Basic algebra problems (10 tasks)
- **calculus_tasks.txt** - Calculus problems (10 tasks)

### Economics
- **microeconomics_tasks.txt** - Microeconomics questions (10 tasks)
- **macroeconomics_tasks.txt** - Macroeconomics questions (10 tasks)
- **global_economics_tasks.txt** - Global/international economics questions (10 tasks)

## File Format

Each file contains tasks in the format:
```
task_id | question | estimated_difficulty | estimated_time(seconds)
```

Example:
```
calculus_001 | Find the derivative of f(x) = 3x² + 5x - 2 | 2 | 45
```

## How to Use

### 1. Train the model on topic-specific data

```bash
python3 cli.py train --data-file data/topic_training_data.csv
```

### 2. Make predictions for specific tasks

For tasks in the training data:
```bash
# Calculus task
python3 cli.py predict --user-id user_001 --item-id calculus_000 --topic calculus

# Microeconomics task
python3 cli.py predict --user-id user_005 --item-id microeconomics_012 --topic microeconomics
```

For NEW tasks (not in training data), use the task files:
```bash
# From algebra_tasks.txt - task algebra_001: "Solve for x: 2x + 5 = 15"
# Estimated difficulty: 1, Time: 30s
# Convert to item features: easy task (b=-1), medium discrimination (a=1.2), quick (beta=3.4)
python3 cli.py predict --user-id user_010 --item-id algebra_001 \
  --topic algebra \
  --item-features '{"a": 1.2, "b": -1.0, "beta": 3.4}'

# From calculus_tasks.txt - task calculus_007: "Find critical points..."
# Estimated difficulty: 4 (hard), Time: 120s
# Convert: hard (b=1.5), good discrimination (a=2.0), longer (beta=4.8)
python3 cli.py predict --user-id user_020 --item-id calculus_007 \
  --topic calculus \
  --item-features '{"a": 2.0, "b": 1.5, "beta": 4.8}'
```

### 3. View statistics for a specific topic

```bash
# All calculus items
python3 cli.py stats --topic calculus --hardest-items 5

# All microeconomics items
python3 cli.py stats --topic microeconomics --hardest-items 5
```

## Converting Estimated Values to Item Features

Use these guidelines to convert estimated difficulty/time to model parameters:

### Difficulty Level → b (difficulty parameter)
- Difficulty 1 (very easy): b = -1.0 to -0.5
- Difficulty 2 (easy): b = -0.5 to 0.0
- Difficulty 3 (medium): b = 0.0 to 0.5
- Difficulty 4 (hard): b = 0.5 to 1.5
- Difficulty 5 (very hard): b = 1.5 to 2.5

### Estimated Time (seconds) → beta (time intensity parameter)
- < 40s: beta = 3.0 to 3.5
- 40-70s: beta = 3.5 to 4.2
- 70-120s: beta = 4.2 to 4.8
- 120-180s: beta = 4.8 to 5.2
- > 180s: beta = 5.2 to 5.8

### Discrimination (a parameter)
- Use a = 1.0 to 1.5 for most tasks
- Use a = 1.5 to 2.5 for tasks that strongly differentiate ability

## Example Workflow

```bash
# 1. Train on topic data
python3 cli.py train --data-file data/topic_training_data.csv

# 2. Test predictions on training data items
python3 cli.py predict --user-id user_000 --item-id calculus_000 --topic calculus

# 3. Test prediction on a NEW calculus problem from the task file
# Task: "Find the second derivative of f(x) = x⁴ - 2x³" (difficulty: 3, time: 70s)
python3 cli.py predict --user-id user_000 --item-id calculus_005_new \
  --topic calculus \
  --item-features '{"a": 1.5, "b": 0.3, "beta": 4.2}'

# 4. Compare predictions across topics for the same user
python3 cli.py predict --user-id user_000 --item-id algebra_010 --topic algebra
python3 cli.py predict --user-id user_000 --item-id calculus_010 --topic calculus

# 5. View topic-specific statistics
python3 cli.py stats --topic calculus --hardest-items 10
python3 cli.py stats --topic microeconomics --hardest-items 10
```

## Testing Different Users

The model has 60 users (user_000 to user_059) with varying abilities. Try different users to see how predictions change:

```bash
# Easy task for different users
python3 cli.py predict --user-id user_000 --item-id algebra_001 --topic algebra
python3 cli.py predict --user-id user_030 --item-id algebra_001 --topic algebra
python3 cli.py predict --user-id user_059 --item-id algebra_001 --topic algebra
```

You should see different probabilities and times based on each user's ability and speed parameters!
