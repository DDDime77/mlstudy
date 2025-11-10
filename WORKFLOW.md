# SmartStudy LNIRT Prediction System - Workflow Documentation

## Overview

This system uses **LNIRT (Lognormal Item Response Theory)** to predict both correctness probability and expected response time for educational tasks. It implements **automatic personalized training** that adapts to each user's performance immediately after they complete tasks.

## Key Principle

**"There is no scenario in which actual data for a user exists but the model isn't personally trained for this user."**

Once a user completes their first task, the system automatically trains a personalized model for them. Every subsequent task completion refines their personalized parameters.

---

## System Architecture

### 1. Database Schema

#### `predictions` table
Stores both predicted and actual values for each task:
```sql
CREATE TABLE predictions (
    task_id INTEGER PRIMARY KEY,
    user_id TEXT,
    topic TEXT,
    difficulty INTEGER,
    predicted_correct REAL,      -- What model predicted
    predicted_time REAL,          -- What model predicted
    actual_correct INTEGER,       -- What actually happened
    actual_time REAL,            -- What actually happened
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

#### `training_data` table
Stores actual results for general training:
```sql
CREATE TABLE training_data (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    topic TEXT,
    difficulty INTEGER,
    correct INTEGER,
    response_time REAL,
    timestamp TIMESTAMP,
    used_for_general_training INTEGER DEFAULT 0
)
```

### 2. Model Architecture

#### General Model Parameters (per topic)
Learned from **all users** collective data:
- **Difficulty parameters** (per difficulty level 1-3):
  - `a`: Discrimination (how well the item differentiates ability)
  - `b`: Difficulty (IRT difficulty parameter)
  - `β`: Time intensity (log-scale time requirement)

#### User-Specific Parameters (per user per topic)
Learned from **individual user's** data:
- `θ` (theta): User ability
- `τ` (tau): User speed

---

## Two User Scenarios

### Scenario 1: Brand New User (No Data)

**Characteristics:**
- User ID not in database
- No prediction history
- No actual task completions

**Prediction Strategy:**
- Uses **population average** parameters
- θ = average of all trained users' θ values
- τ = average of all trained users' τ values
- Uses general difficulty parameters (a, b, β)

**Example:**
```bash
python3 smart_cli.py predict --user-id new_student --topic calculus --difficulty 2 --save
# Uses population average: θ=-0.061, τ=-0.040
# Prediction: P(correct)=32.4%, time=121.8s
```

### Scenario 2: Existing User (Has Data + Automatic Training)

**Characteristics:**
- User has completed at least 1 task
- Has predicted+actual data pairs in database
- **Automatically trained** personalized model

**Prediction Strategy:**
- Uses **personalized** parameters (θ, τ)
- Parameters learned from user's actual performance
- Error-aware: corrects for systematic prediction biases
- Uses general difficulty parameters (a, b, β)

**Example:**
```bash
# After user completes first task and automatic training occurs
python3 smart_cli.py predict --user-id new_student --topic calculus --difficulty 2
# Uses personalized parameters: θ=3.000, τ=0.427
# Prediction: P(correct)=79.5%, time=76.4s
```

---

## Complete Workflow

### Initial Setup (One-Time)

```bash
# 1. Load CSV training data (historical data)
python3 load_csv_to_predictions.py data/ib/calculus.csv calculus --force

# 2. Train general model on collective data
python3 smart_cli.py train --topic calculus
```

### User Workflow (Repeating Cycle)

#### Step 1: Make Prediction

```bash
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

**What happens:**
- System checks if `alice` has personalized parameters
- **If no:** Uses population average (Scenario 1)
- **If yes:** Uses personalized parameters (Scenario 2)
- Calculates P(correct) and expected time
- Saves prediction to database with `task_id`

**Output:**
```
User: alice
Topic: calculus
Difficulty: 2

PREDICTION RESULTS
Probability of Correct: 32.4%
Expected Time: 121.8 seconds (2.0 minutes)

✓ Prediction saved with task_id=1234
  Update later with: python3 smart_cli.py update --task-id 1234 --correct [0/1] --time [seconds]
```

#### Step 2: User Completes Task

User answers the question and system records:
- Correctness: 1 (correct) or 0 (incorrect)
- Time taken: actual seconds

#### Step 3: Update with Actual Results (AUTOMATIC TRAINING)

```bash
python3 smart_cli.py update --task-id 1234 --correct 1 --time 95.5
```

**What happens (FULLY AUTOMATIC):**

1. **Database Update:**
   - Stores actual_correct=1, actual_time=95.5 in predictions table
   - Adds actual results to training_data table

2. **AUTOMATIC USER-SPECIFIC TRAINING:**
   - Retrieves ALL completed tasks for this user (predicted+actual pairs)
   - Runs error-aware LNIRT training:
     - Analyzes prediction errors (predicted vs actual)
     - Detects systematic biases
     - Optimizes user parameters (θ, τ) using maximum likelihood
     - Applies bias correction
   - Saves personalized model to disk
   - User now in Scenario 2 (personalized)

3. **General Training Pool:**
   - Actual results added to training pool
   - Optional: Retrain general model if threshold met (e.g., every 50 new responses)

**Output:**
```
UPDATE PREDICTION WITH ACTUAL RESULTS
Task ID: 1234
User: alice
Predicted:
  Correctness: 32.4%
  Time: 121.8s
Actual:
  Correctness: CORRECT
  Time: 95.5s

✓ Database updated

AUTOMATIC USER-SPECIFIC TRAINING
Training personalized model for alice...
  Using 1 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.676 (positive = actual better than predicted)
  Time bias (log): -0.360 (positive = actual slower than predicted)
  ⚠ Model systematically UNDERESTIMATES user ability
  ⚠ Model systematically OVERESTIMATES time needed

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 3.000
    Speed (τ): 0.427

✓ Personalized model updated for alice
✓ Actual results added to general training pool

UPDATE COMPLETE!
```

#### Step 4: Next Prediction (Uses Personalized Model)

```bash
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

**Now uses personalized parameters:**
```
Prediction: P(correct)=79.5%, time=76.4s
```

Much better than initial 32.4% / 121.8s!

---

## Error-Aware Training Details

### What Makes It "Error-Aware"?

Traditional IRT only uses actual results. Our system uses **BOTH** predicted and actual values to detect and correct systematic biases.

### Error Analysis

For each user, the system analyzes:

1. **Correctness Bias:**
   ```python
   correctness_error = actual_correct - predicted_correct
   bias = mean(correctness_errors)
   ```
   - Positive bias: Model underestimates user ability
   - Negative bias: Model overestimates user ability

2. **Time Bias (Log Scale):**
   ```python
   time_error_log = log(actual_time) - log(predicted_time)
   bias_log = mean(time_error_log)
   ```
   - Positive bias: User slower than predicted
   - Negative bias: User faster than predicted

### Parameter Optimization

Uses **Maximum Likelihood Estimation** with error-awareness:

```python
def fit_user_specific(user_data, user_id):
    # Step 1: Analyze prediction errors
    error_stats = analyze_prediction_errors(user_data)

    # Step 2: Define error-aware likelihood
    def error_aware_likelihood(params):
        theta, tau = params

        # Standard LNIRT likelihood
        log_likelihood = calculate_lnirt_likelihood(theta, tau, user_data)

        # Error-aware penalty
        if abs(error_stats['correctness_bias']) > 0.15:
            # Penalize parameters that would produce similar errors
            penalty = error_penalty(theta, tau, error_stats)
            log_likelihood += penalty

        return -log_likelihood

    # Step 3: Optimize with scipy
    result = minimize(error_aware_likelihood, initial_params,
                     method='L-BFGS-B', bounds=[(-3, 3), (-3, 3)])

    # Step 4: Apply bias correction
    theta_new, tau_new = result.x
    if abs(error_stats['correctness_bias']) > 0.15:
        correction = error_stats['correctness_bias'] * 0.5
        theta_new += correction

    return theta_new, tau_new
```

---

## CLI Commands

### `predict` - Make Prediction

```bash
python3 smart_cli.py predict --user-id <user> --topic <topic> --difficulty <1|2|3> [--save]
```

**Options:**
- `--user-id`: User identifier
- `--topic`: Topic name (e.g., calculus, algebra)
- `--difficulty`: Difficulty level (1=easy, 2=medium, 3=hard)
- `--save`: Save prediction to database (required for update workflow)

**Example:**
```bash
python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

### `update` - Update with Actual Results (AUTOMATIC TRAINING)

```bash
python3 smart_cli.py update --task-id <id> --correct <0|1> --time <seconds> [--retrain-threshold <N>]
```

**Options:**
- `--task-id`: Task ID from prediction (required)
- `--correct`: 1 if correct, 0 if incorrect
- `--time`: Actual time in seconds
- `--retrain-threshold`: Optional, trigger general retraining after N new responses (default: 50)

**Example:**
```bash
python3 smart_cli.py update --task-id 1234 --correct 1 --time 95.5
```

**IMPORTANT:** This automatically triggers user-specific training!

### `train` - Train Model

```bash
python3 smart_cli.py train --topic <topic> [--data-file <csv>] [--user-id <user>] [--stats]
```

**Options:**
- `--topic`: Topic name (required)
- `--data-file`: Optional CSV file for initial training
- `--user-id`: Manual user-specific training (NOTE: happens automatically on update)
- `--stats`: Show statistics after training

**Example (General Training):**
```bash
python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv --stats
```

**Example (Manual User-Specific Training):**
```bash
python3 smart_cli.py train --topic calculus --user-id alice
```

Note: Manual user-specific training is rarely needed since it happens automatically on update.

### `stats` - Display Statistics

```bash
python3 smart_cli.py stats [--topic <topic>]
```

**Options:**
- `--topic`: Show stats for specific topic (omit to list all topics)

**Example:**
```bash
python3 smart_cli.py stats --topic calculus
```

---

## Mathematical Details

### LNIRT Model

#### Correctness: 2-Parameter Logistic IRT

```
P(correct | θ, a, b) = 1 / (1 + exp(-a * (θ - b)))
```

Where:
- θ: User ability
- a: Item discrimination
- b: Item difficulty

#### Response Time: Lognormal Distribution

```
log(RT) ~ N(β - τ, σ²)
E[RT] = exp(β - τ)
```

Where:
- τ: User speed (higher = faster)
- β: Item time intensity
- σ: Response time variability (typically fixed at 1.0)

#### Joint Likelihood

```
L(θ, τ, a, b, β | data) = ∏ P(correct_i | θ, a, b) * P(RT_i | τ, β, σ)
```

Optimized using scipy.optimize.minimize with L-BFGS-B method.

---

## Testing

### Run All Tests

```bash
# Comprehensive error analysis (31 tests)
python3 comprehensive_error_analysis.py

# Automatic training workflow test (11 tests)
python3 test_automatic_training.py

# Three user scenarios test
python3 test_three_scenarios.py
```

### Expected Results

All tests should pass:
- ✓ Database integrity (1500 records, all have predicted+actual)
- ✓ Model consistency (50 users, parameters within bounds)
- ✓ Prediction correctness
- ✓ Error-aware training functionality
- ✓ Automatic training workflow
- ✓ All users with data are trained

---

## Key Insights

### 1. Automatic Personalization

- **No manual training needed** - happens automatically on update
- **Immediate adaptation** - user's first task completion triggers personalization
- **Continuous refinement** - parameters improve with each task

### 2. Two-Tier Parameter System

**General Parameters (a, b, β):**
- Learned from all users' collective data
- Ensures objective difficulty calibration
- Same for all users

**User Parameters (θ, τ):**
- New user: Population average
- Existing user: Personalized via automatic training
- Unique to each user

### 3. Error-Aware Learning

Unlike traditional IRT:
- Analyzes **prediction errors** (predicted vs actual)
- Detects **systematic biases** (consistently over/under-predicting)
- Corrects parameters to **fix those biases**
- Results in **more accurate** future predictions

### 4. Real SmartStudy Workflow

```
predict → user answers → update → automatic training → next prediction (personalized)
   ↓                          ↓                              ↓
  save                    record actual              uses personalized params
```

Every user follows this cycle. No exceptions. No user has actual data without training.

---

## File Structure

```
ml_lnirt_playground/
├── smart_cli.py                      # Main CLI interface
├── topic_lnirt.py                    # LNIRT model implementation
├── predictions_db.py                 # Database interface
├── load_csv_to_predictions.py        # CSV data loader
├── test_automatic_training.py        # Automatic training test
├── comprehensive_error_analysis.py   # 31 comprehensive tests
├── test_three_scenarios.py          # Scenario comparison test
├── predictions.db                    # SQLite database
├── models/
│   └── calculus.pkl                 # Trained model (50 users)
├── data/
│   └── ib/
│       └── calculus.csv             # Training data (1500 records)
└── WORKFLOW.md                       # This file
```

---

## FAQ

**Q: What happens on a user's first prediction?**

A: System uses population average parameters (θ, τ are averages of all trained users).

**Q: When does personalized training happen?**

A: Automatically, immediately after first actual result is recorded via `update` command.

**Q: Can a user have actual data without personalized training?**

A: No. The system ensures this never happens. Update command always triggers automatic training.

**Q: What if I want to manually retrain a user?**

A: Use `python3 smart_cli.py train --topic <topic> --user-id <user>`, but this is rarely needed.

**Q: How many tasks needed for good personalization?**

A: Even 1 task provides personalization. Accuracy improves with more tasks (typically stabilizes around 5-10 tasks).

**Q: What if predictions are consistently wrong?**

A: Error-aware training detects and corrects systematic biases automatically. More data → better correction.

**Q: When does general model get retrained?**

A: Optionally after N new responses (default 50). Configure with `--retrain-threshold` in update command.

---

## Performance

**Automatic Training Speed:**
- 1 task: ~0.1 seconds
- 10 tasks: ~0.3 seconds
- 30 tasks: ~0.8 seconds

**Prediction Speed:**
- < 0.01 seconds per prediction

**Database Size:**
- 1500 records ≈ 200 KB
- 50 users trained ≈ 50 KB model file

---

## Credits

Built with:
- **Python 3.10**
- **NumPy** for numerical computation
- **Pandas** for data manipulation
- **SciPy** for optimization
- **SQLite** for database
- **LNIRT** mathematical framework

---

## Version History

- **v2.0** (Current): Automatic user-specific training, error-aware learning
- **v1.0**: Basic LNIRT with manual training
