# LNIRT Model Playground

Machine Learning playground for testing the **LNIRT (Joint IRT + Lognormal Response Time) Model** for the smartstudy project.

## Overview

This implementation provides a sophisticated machine learning model that simultaneously predicts:
1. **Probability of correct response** - Whether a user will answer a question correctly
2. **Expected response time** - How long a user will take to complete a task

The model uses **Item Response Theory (IRT)** combined with a **lognormal response time model** to create joint predictions that account for both user ability/speed and task difficulty/complexity.

## What is LNIRT?

LNIRT combines two well-established psychometric models:

### 1. Item Response Theory (IRT) - 2PL Model
Predicts the probability of a correct response based on:
- **θ (theta)** - User ability (higher = more capable)
- **a** - Item discrimination (how well the item distinguishes between high/low ability)
- **b** - Item difficulty (higher = harder question)

**Formula:** P(correct | θ, a, b) = 1 / (1 + exp(-a(θ - b)))

### 2. Lognormal Response Time Model
Models response time as log-normally distributed:
- **τ (tau)** - User speed (higher = faster responses)
- **β (beta)** - Item time intensity (higher = takes longer)
- **σ (sigma)** - Residual variance

**Formula:** log(RT) ~ N(β - τ, σ²)

### Why Joint Modeling?
By modeling both correctness and time together, the LNIRT model captures the relationship between ability and speed, providing more accurate predictions than either model alone.

## Project Structure

```
ml_lnirt_playground/
├── cli.py                          # Command-line interface
├── lnirt_model.py                  # LNIRT model implementation
├── generate_sample_data.py         # Sample data generator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── sample_training_data.csv    # Generated training data (1500 samples)
│   ├── item_bank.json              # Question bank with metadata
│   └── user_data_template.csv      # Template for your own data
└── models/
    └── lnirt_model.pkl             # Trained model (created after training)
```

## Installation

1. Install required Python packages:
```bash
pip3 install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data (Already Done)

Sample data has been pre-generated with:
- 50 users
- 100 items/questions
- 1500 total responses
- Topics: algebra, geometry, calculus, statistics, trigonometry

To regenerate:
```bash
python3 generate_sample_data.py
```

### 2. Train the Model

Train the model on the sample data:
```bash
python3 cli.py train --data-file data/sample_training_data.csv --stats
```

Options:
- `--data-file` / `-d`: Path to training data CSV (required)
- `--model-file` / `-m`: Where to save the model (default: models/lnirt_model.pkl)
- `--max-iter`: Maximum optimization iterations (default: 1000)
- `--stats`: Display detailed statistics after training
- `--verbose` / `-v`: Show detailed training progress

### 3. Make Predictions

Predict for a user on a specific task:
```bash
python3 cli.py predict --user-id user_001 --item-id item_050
```

The output will show:
- Probability of correct answer (0-100%)
- Expected response time (in seconds and minutes)
- Interpretation of the predictions
- Warnings if user/item is not in training data

For a new item (not in training data), provide estimated features:
```bash
python3 cli.py predict --user-id user_001 --item-id new_task \
  --item-features '{"a": 1.5, "b": 0.5, "beta": 4.0}'
```

### 4. View Model Statistics

Display comprehensive model statistics:
```bash
python3 cli.py stats --top-users 10 --hardest-items 10
```

This shows:
- User ability and speed distributions
- Item difficulty and discrimination parameters
- Top performing users
- Hardest items

## Data Format

### Training Data Format

Your training data must be a CSV file with these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | string | Unique user identifier | "user_001" |
| `item_id` | string | Unique task/question identifier | "item_050" |
| `correct` | integer | 1 = correct, 0 = incorrect | 1 |
| `response_time` | float | Time in seconds | 45.2 |

**Example:**
```csv
user_id,item_id,correct,response_time
user_001,item_001,1,45.2
user_001,item_002,0,120.5
user_002,item_001,1,30.8
```

See `data/user_data_template.csv` for a complete example.

## Sample Datasets Explanation

### 1. `data/sample_training_data.csv`
**Purpose:** Pre-generated training data to test the model

**Contains:**
- 1500 response records from 50 simulated users
- Each user answered 30 different questions
- Response correctness and times generated using realistic IRT parameters
- Users have varying abilities and speeds
- Items have varying difficulties and time intensities

**Why it's useful:** Allows you to train and test the model immediately without collecting real data. The data follows the same statistical patterns expected in real educational assessments.

### 2. `data/item_bank.json`
**Purpose:** Metadata about available questions/tasks

**Contains:**
- 100 items with unique IDs
- Question text (sample/placeholder)
- Topic classification (algebra, geometry, etc.)
- Estimated difficulty (1-5 scale)

**Why it's useful:** Demonstrates how to organize question metadata. In production, this would contain actual question content and could be used to select appropriate questions for users.

### 3. `data/user_data_template.csv`
**Purpose:** Template showing the exact format for your own training data

**Contains:**
- Example rows showing the required format
- Comments explaining each column
- Sample values demonstrating correct data types

**Why it's useful:** When you want to train the model on your own data (real user responses), use this template to ensure your data is formatted correctly.

## Use Cases in SmartStudy

### 1. Time Prediction for Users
Before a user starts a task:
```python
p_correct, expected_time = model.predict(user_id, task_id)
print(f"This task will take approximately {expected_time/60:.1f} minutes")
```

### 2. Adaptive Difficulty Selection
After a user completes a task:
```python
p_correct, _ = model.predict(user_id, completed_task_id)
actual_correct = user_got_it_correct()

if actual_correct and p_correct < 0.5:
    # User exceeded expectations - give harder tasks
    next_task = select_harder_task()
elif not actual_correct and p_correct > 0.5:
    # User underperformed - give easier tasks
    next_task = select_easier_task()
```

### 3. Performance Tracking
Monitor user progress over time:
```python
user_stats = model.get_user_stats()
user_row = user_stats[user_stats['user_id'] == current_user]
ability = user_row['ability_theta'].values[0]
# Track how ability changes as user learns
```

## Model Parameters Interpretation

### User Parameters
- **Ability (θ)**: Higher values indicate more capable users
  - Typical range: -3 to +3
  - Mean: 0, SD: 1
- **Speed (τ)**: Higher values indicate faster users
  - Typical range: -2 to +2
  - Mean: 0, SD: 0.5-1

### Item Parameters
- **Discrimination (a)**: How well item distinguishes ability levels
  - Typical range: 0.5 to 2.5
  - Higher = better at discriminating
- **Difficulty (b)**: How hard the item is
  - Typical range: -3 to +3
  - Mean: 0, higher = harder
- **Time Intensity (β)**: Base log-time for item
  - Typical range: 3 to 6 (exp(3)≈20s, exp(6)≈400s)
  - Higher = takes longer

## Advanced Usage

### Training on Your Own Data

1. Collect user response data in the required format
2. Save as CSV with columns: user_id, item_id, correct, response_time
3. Train the model:
```bash
python3 cli.py train --data-file your_data.csv --model-file your_model.pkl
```

### Incremental Learning

To add new data to an existing model, combine old and new data:
```bash
# Combine datasets
cat old_data.csv new_data.csv > combined_data.csv

# Retrain model
python3 cli.py train --data-file combined_data.csv --model-file updated_model.pkl
```

### Predicting for New Users

For users not in the training data, the model uses population average parameters:
- Ability: 0 (average)
- Speed: 0 (average)

Predictions will be less accurate but still useful for initial task selection.

### Predicting for New Items

For new items, you can:
1. Use population averages (automatic if no features provided)
2. Provide estimated features based on expert judgment:
```bash
python3 cli.py predict --user-id user_001 --item-id new_item \
  --item-features '{"a": 1.0, "b": 0.5, "beta": 4.5}'
```

## Troubleshooting

### Training fails with convergence warnings
- Try increasing `--max-iter` to 2000 or higher
- Ensure you have enough data (at least 100-200 responses)
- Check for data quality issues (negative times, invalid correct values)

### Predictions seem inaccurate
- Model needs sufficient training data for each user/item
- Users/items with very few responses have less reliable parameter estimates
- Consider collecting more data or using regularization

### Model file not found
- Make sure you've trained the model first with `train` command
- Check that the `--model-file` path matches between train and predict

## Technical Details

### Optimization
- Method: L-BFGS-B (bounded optimization)
- Objective: Maximum likelihood estimation
- Parameters are estimated jointly for optimal accuracy

### Model Assumptions
1. User ability and speed are stable over time
2. Item parameters are fixed
3. Responses are conditionally independent given user/item parameters
4. Response times follow a lognormal distribution

### Limitations
- Requires sufficient data per user/item (ideally 10+ responses)
- Does not account for learning effects (ability changes over time)
- New users/items have less accurate predictions
- Assumes independence between items (no learning from previous questions)

## Next Steps for Integration

To integrate this model into the smartstudy project:

1. **Data Collection**: Set up a database to store user responses in the required format
2. **Model Training Pipeline**: Regularly retrain the model with new data (e.g., nightly batch job)
3. **API Integration**: Create an API endpoint that loads the model and makes predictions
4. **Real-time Updates**: Implement online learning to update parameters as users complete tasks
5. **Feature Engineering**: Add more sophisticated item features (topic, type, complexity, etc.)

## References

- van der Linden, W. J. (2007). A hierarchical framework for modeling speed and accuracy on test items. *Psychometrika*, 72(3), 287-308.
- Fox, J. P., & Marianti, S. (2016). Joint modeling of ability and differential speed using responses and response times. *Multivariate Behavioral Research*, 51(4), 540-553.

## License

This is a playground implementation for testing purposes. Modify and use as needed for the smartstudy project.
