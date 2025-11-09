# SmartStudy ML System

Machine Learning system for predicting student performance on IB Diploma Programme tasks.

## What This Does

- **Predicts** whether a student will answer correctly
- **Predicts** how long a student will take
- **Learns** from each student's performance to personalize future predictions
- **Separates** each topic completely (calculus ≠ algebra ≠ economics)

## Quick Start

```bash
# 1. Train on calculus
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv

# 2. Predict for a student
python3 smart_cli.py predict --user-id student_001 --topic calculus --difficulty 2 --save

# 3. Update with actual results
python3 smart_cli.py update --task-id 1 --correct 1 --time 180
```

## Features

✅ **Topic-Based Models** - Each subject has its own independent model

✅ **Simple Interface** - Predict using only: user_id + topic + difficulty (1-3)

✅ **Dual Learning**:
  - General: Learns from all users
  - Personalized: Adapts to individual performance

✅ **Database Tracking** - All predictions saved in SQLite

✅ **IB DP Aligned** - Training data for Math AA & Economics

## CLI Commands

### Train
```bash
python3 smart_cli.py train --topic <topic> --data-file <csv>
```

### Predict
```bash
python3 smart_cli.py predict --user-id <user> --topic <topic> --difficulty <1|2|3> --save
```

### Update
```bash
python3 smart_cli.py update --task-id <id> --correct <0|1> --time <seconds>
```

### Stats
```bash
python3 smart_cli.py stats [--topic <topic>]
```

## Documentation

See **SMARTSTUDY_README.md** for complete documentation.

## Requirements

```bash
pip3 install -r requirements.txt
```
