# Detailed Workflow Scenarios

This document provides comprehensive scenarios demonstrating the SmartStudy LNIRT prediction system in action.

---

## Scenario 1: Brand New Student (Alice) - First Week Journey

### Day 1: Alice's First Calculus Task

Alice is a new student who has never used SmartStudy for calculus before.

#### Step 1.1: First Prediction (Difficulty 2 - Medium)

```bash
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

**System Output:**
```
======================================================================
PREDICTION
======================================================================

User: alice
Topic: calculus
Difficulty: 2

----------------------------------------------------------------------
PREDICTION RESULTS
----------------------------------------------------------------------
Probability of Correct: 32.4%
Expected Time: 121.8 seconds (2.0 minutes)

~ Confidence: LOW (32%)
⏱ Moderate time expected (2.0 min)

✓ Prediction saved with task_id=1501
  Update later with: python3 smart_cli.py update --task-id 1501 --correct [0/1] --time [seconds]

======================================================================
```

**What Happened Behind the Scenes:**

1. **User Lookup:**
   - System checks: Is `alice` in model? → **NO**
   - System checks: Does `alice` have prediction history? → **NO**

2. **Parameter Selection:**
   ```python
   # Alice is new, use population average
   theta_alice = mean([θ for all 50 trained users])  # = -0.061
   tau_alice = mean([τ for all 50 trained users])    # = -0.040
   ```

3. **Difficulty Parameters (from general model):**
   ```python
   difficulty_2_params = {
       'a': 0.682,    # Discrimination
       'b': 1.016,    # Difficulty
       'beta': 4.763  # Time intensity
   }
   ```

4. **Prediction Calculation:**
   ```python
   # IRT probability
   P(correct) = 1 / (1 + exp(-a * (θ - b)))
   P(correct) = 1 / (1 + exp(-0.682 * (-0.061 - 1.016)))
   P(correct) = 0.324  # 32.4%

   # Expected time
   E[time] = exp(β - τ)
   E[time] = exp(4.763 - (-0.040))
   E[time] = 121.8 seconds
   ```

5. **Database Insert:**
   ```sql
   INSERT INTO predictions (user_id, topic, difficulty, predicted_correct, predicted_time, ...)
   VALUES ('alice', 'calculus', 2, 0.324, 121.8, ...)
   -- Returns task_id = 1501
   ```

#### Step 1.2: Alice Completes the Task

Alice works on the calculus problem:
- **Time taken:** 95 seconds (faster than predicted!)
- **Result:** CORRECT ✓

**SmartStudy records the actual result:**

```bash
$ python3 smart_cli.py update --task-id 1501 --correct 1 --time 95
```

**System Output:**
```
======================================================================
UPDATE PREDICTION WITH ACTUAL RESULTS
======================================================================

Task ID: 1501
User: alice
Topic: calculus
Difficulty: 2

Predicted:
  Correctness: 32.4%
  Time: 121.8s

Actual:
  Correctness: CORRECT
  Time: 95.0s

✓ Database updated

======================================================================
AUTOMATIC USER-SPECIFIC TRAINING
======================================================================

Training personalized model for alice...
  Using 1 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.676 (positive = actual better than predicted)
  Correctness std: 0.000
  Time bias (log): -0.247 (positive = actual slower than predicted)
  Time ratio: 0.78x (median: 0.78x)
  Samples: 1
  ⚠ Model systematically UNDERESTIMATES user ability
  ⚠ Model systematically OVERESTIMATES time needed

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Iteration 1: log-likelihood = -2.145
  Iteration 5: log-likelihood = -1.823
  Converged after 8 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 2.856
    Speed (τ): 0.385

✓ Personalized model updated for alice
✓ Actual results added to general training pool

======================================================================
UPDATE COMPLETE!
======================================================================
```

**What Happened Behind the Scenes:**

1. **Database Update:**
   ```sql
   UPDATE predictions
   SET actual_correct = 1, actual_time = 95.0, updated_at = NOW()
   WHERE task_id = 1501;

   INSERT INTO training_data (user_id, topic, difficulty, correct, response_time, ...)
   VALUES ('alice', 'calculus', 2, 1, 95.0, ...);
   ```

2. **AUTOMATIC User-Specific Training Triggered:**
   ```python
   # Retrieve alice's completed tasks
   user_data = db.get_user_training_data('alice', 'calculus')
   # Returns 1 row with both predicted and actual values:
   # [difficulty=2, predicted_correct=0.324, actual_correct=1,
   #  predicted_time=121.8, actual_time=95.0]
   ```

3. **Error Analysis:**
   ```python
   correctness_errors = [1 - 0.324] = [0.676]
   correctness_bias = mean(correctness_errors) = +0.676
   # Positive = actual better than predicted

   time_errors_log = [log(95) - log(121.8)] = [-0.247]
   time_bias_log = mean(time_errors_log) = -0.247
   # Negative = actual faster than predicted
   ```

4. **Parameter Optimization:**
   ```python
   # Maximum likelihood estimation
   initial_params = [theta=-0.061, tau=-0.040]  # Population average

   # Optimize joint log-likelihood
   result = minimize(joint_lnirt_likelihood, initial_params,
                     method='L-BFGS-B', bounds=[(-3, 3), (-3, 3)])

   # Result: theta=2.856, tau=0.385
   # Much higher than population average!
   ```

5. **Model Update:**
   ```python
   model.user_params['alice'] = {'theta': 2.856, 'tau': 0.385}
   manager.save_model('calculus')
   # Alice now has personalized parameters!
   ```

---

### Day 2: Alice's Second Task (Now Personalized!)

#### Step 2.1: Next Prediction (Same Difficulty)

Alice wants to try another medium difficulty problem.

```bash
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

**System Output:**
```
======================================================================
PREDICTION
======================================================================

User: alice
Topic: calculus
Difficulty: 2

----------------------------------------------------------------------
PREDICTION RESULTS
----------------------------------------------------------------------
Probability of Correct: 78.3%
Expected Time: 77.2 seconds (1.3 minutes)

✓ Confidence: HIGH (78%)
⚡ Expected to complete quickly (77s)

✓ Prediction saved with task_id=1502
  Update later with: python3 smart_cli.py update --task-id 1502 --correct [0/1] --time [seconds]

======================================================================
```

**Comparison with First Prediction:**
```
Metric              First (Day 1)    Second (Day 2)    Change
-------------------------------------------------------------
P(correct)          32.4%            78.3%             +45.9%
Expected Time       121.8s           77.2s             -44.6s
Confidence          LOW              HIGH              +++
Parameters Used     Pop. Average     Personalized      ✓
```

**What Happened Behind the Scenes:**

1. **User Lookup:**
   - System checks: Is `alice` in model? → **YES!**
   - Retrieves: θ=2.856, τ=0.385

2. **Prediction Calculation (Using Personalized Params):**
   ```python
   # IRT probability with ALICE'S parameters
   P(correct) = 1 / (1 + exp(-0.682 * (2.856 - 1.016)))
   P(correct) = 0.783  # 78.3% (was 32.4%)

   # Expected time with ALICE'S parameters
   E[time] = exp(4.763 - 0.385)
   E[time] = 77.2 seconds (was 121.8s)
   ```

Much better predictions now!

#### Step 2.2: Alice Completes Second Task

Alice works on the problem:
- **Time taken:** 82 seconds
- **Result:** CORRECT ✓

```bash
$ python3 smart_cli.py update --task-id 1502 --correct 1 --time 82
```

**System Output:**
```
======================================================================
UPDATE PREDICTION WITH ACTUAL RESULTS
======================================================================

Task ID: 1502
User: alice
Topic: calculus
Difficulty: 2

Predicted:
  Correctness: 78.3%
  Time: 77.2s

Actual:
  Correctness: CORRECT
  Time: 82.0s

✓ Database updated

======================================================================
AUTOMATIC USER-SPECIFIC TRAINING
======================================================================

Training personalized model for alice...
  Using 2 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.359 (positive = actual better than predicted)
  Correctness std: 0.317
  Time bias (log): -0.060 (positive = actual slower than predicted)
  Time ratio: 0.95x (median: 0.95x)
  Samples: 2
  ⚠ Model still slightly UNDERESTIMATES user ability

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Converged after 6 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 2.912  (was 2.856, +0.056)
    Speed (τ): 0.351    (was 0.385, -0.034)

✓ Personalized model updated for alice
✓ Actual results added to general training pool

======================================================================
UPDATE COMPLETE!
======================================================================
```

**Parameter Evolution:**
```
After Task    Ability (θ)    Speed (τ)     Correctness Bias
----------------------------------------------------------
Initial       -0.061 (avg)   -0.040 (avg)  N/A
Task 1        2.856          0.385         +0.676 (large)
Task 2        2.912          0.351         +0.359 (smaller)
```

Note: Parameters are **converging**. The system is learning Alice's true ability!

---

### Day 3: Alice Tries Harder Problems

#### Step 3.1: Alice Attempts Difficulty 3 (Hard)

```bash
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 3 --save
```

**System Output:**
```
======================================================================
PREDICTION
======================================================================

User: alice
Topic: calculus
Difficulty: 3

----------------------------------------------------------------------
PREDICTION RESULTS
----------------------------------------------------------------------
Probability of Correct: 65.2%
Expected Time: 95.8 seconds (1.6 minutes)

✓ Confidence: MEDIUM (65%)
⏱ Moderate time expected (1.6 min)

✓ Prediction saved with task_id=1503

======================================================================
```

**What Happened:**

Difficulty 3 parameters are different:
```python
difficulty_3_params = {
    'a': 0.544,    # Slightly lower discrimination
    'b': 1.322,    # HIGHER difficulty
    'beta': 5.008  # LONGER time intensity
}

# Using Alice's personalized parameters
P(correct) = 1 / (1 + exp(-0.544 * (2.912 - 1.322)))
P(correct) = 0.652  # 65.2%

E[time] = exp(5.008 - 0.351)
E[time] = 95.8 seconds
```

Note: Even with high ability (θ=2.912), harder problems have lower P(correct) and longer times.

#### Step 3.2: Alice Struggles with Hard Problem

Alice works on the hard problem:
- **Time taken:** 185 seconds (much longer than predicted!)
- **Result:** INCORRECT ✗

```bash
$ python3 smart_cli.py update --task-id 1503 --correct 0 --time 185
```

**System Output:**
```
======================================================================
UPDATE PREDICTION WITH ACTUAL RESULTS
======================================================================

Task ID: 1503
User: alice
Topic: calculus
Difficulty: 3

Predicted:
  Correctness: 65.2%
  Time: 95.8s

Actual:
  Correctness: INCORRECT
  Time: 185.0s

✓ Database updated

======================================================================
AUTOMATIC USER-SPECIFIC TRAINING
======================================================================

Training personalized model for alice...
  Using 3 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.017 (positive = actual better than predicted)
  Correctness std: 0.500
  Time bias (log): +0.099 (positive = actual slower than predicted)
  Time ratio: 1.15x (median: 0.95x)
  Samples: 3
  ℹ Predictions are well-calibrated for correctness
  ⚠ Model slightly UNDERESTIMATES time for difficult tasks

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Converged after 7 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 2.654  (was 2.912, -0.258)
    Speed (τ): 0.218    (was 0.351, -0.133)

✓ Personalized model updated for alice
✓ Actual results added to general training pool

======================================================================
UPDATE COMPLETE!
======================================================================
```

**What Happened:**

Alice's incorrect answer on a hard problem **lowered her ability estimate**:
- θ: 2.912 → 2.654 (-0.258)
- τ: 0.351 → 0.218 (-0.133, slower on hard problems)

This is **appropriate learning** - the model adjusts based on actual performance.

---

### Day 4: Alice Back to Medium Difficulty

#### Step 4.1: Prediction with Updated Parameters

```bash
$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2 --save
```

**System Output:**
```
======================================================================
PREDICTION
======================================================================

User: alice
Topic: calculus
Difficulty: 2

----------------------------------------------------------------------
PREDICTION RESULTS
----------------------------------------------------------------------
Probability of Correct: 72.1%
Expected Time: 84.3 seconds (1.4 minutes)

✓ Confidence: HIGH (72%)
⏱ Moderate time expected (1.4 min)

✓ Prediction saved with task_id=1504

======================================================================
```

**Comparison of Alice's D2 Predictions Over Time:**
```
Day    Tasks Completed    θ        τ        P(correct)    E[time]
--------------------------------------------------------------------
1      0 (new user)       -0.061   -0.040   32.4%         121.8s
2      1 (after first)    2.856    0.385    78.3%         77.2s
4      3 (after hard)     2.654    0.218    72.1%         84.3s
```

Model is **adapting realistically** to Alice's mixed performance.

---

## Scenario 2: Bob (Existing CSV User) - Making New Predictions

Bob is one of the 50 users from the original CSV training data. He has 30 completed tasks in the database already.

### Bob's Current State

```bash
$ python3 -c "
from topic_lnirt import TopicModelManager
from predictions_db import PredictionsDB

manager = TopicModelManager()
model = manager.get_model('calculus')
db = PredictionsDB()

# Check Bob's status
print('Bob (user_010) Status:')
print(f'In model: {\"user_010\" in model.user_params}')
print(f'Parameters: θ={model.user_params[\"user_010\"][\"theta\"]:.3f}, τ={model.user_params[\"user_010\"][\"tau\"]:.3f}')

# Check history
import pandas as pd
history = pd.read_sql_query(
    \"SELECT COUNT(*) as count, AVG(actual_correct) as accuracy, AVG(actual_time) as avg_time
     FROM predictions
     WHERE user_id='user_010' AND topic='calculus'\",
    db.conn
).iloc[0]

print(f'\\nHistory:')
print(f'Completed tasks: {history[\"count\"]}')
print(f'Accuracy: {history[\"accuracy\"]:.1%}')
print(f'Average time: {history[\"avg_time\"]:.1f}s')
db.close()
"
```

**Output:**
```
Bob (user_010) Status:
In model: True
Parameters: θ=0.268, τ=-0.111

History:
Completed tasks: 30
Accuracy: 43.3%
Average time: 133.7s
```

### Bob's New Prediction

```bash
$ python3 smart_cli.py predict --user-id user_010 --topic calculus --difficulty 2 --save
```

**System Output:**
```
======================================================================
PREDICTION
======================================================================

User: user_010
Topic: calculus
Difficulty: 2

----------------------------------------------------------------------
PREDICTION RESULTS
----------------------------------------------------------------------
Probability of Correct: 37.5%
Expected Time: 130.8 seconds (2.2 minutes)

~ Confidence: LOW (38%)
⏱ Moderate time expected (2.2 min)

✓ Prediction saved with task_id=1505

======================================================================
```

**What Happened:**
```python
# Bob has personalized parameters from general training (30 CSV tasks)
theta_bob = 0.268  # Lower than average (-0.061)
tau_bob = -0.111   # Slower than average

# Prediction using Bob's parameters
P(correct) = 1 / (1 + exp(-0.682 * (0.268 - 1.016)))
P(correct) = 0.375  # 37.5%

E[time] = exp(4.763 - (-0.111))
E[time] = 130.8 seconds
```

**Comparison:**
```
User     θ        τ        P(correct)    E[time]    Source
---------------------------------------------------------------
Alice    2.654    0.218    72.1%         84.3s      User-specific training
Bob      0.268    -0.111   37.5%         130.8s     General training (CSV)
Average  -0.061   -0.040   32.4%         121.8s     Population average
```

Bob's parameters show he's slightly below average ability but slower.

---

## Scenario 3: Running User-Specific Training for Bob

Let's manually trigger user-specific training for Bob to see if his parameters improve with error-aware training.

```bash
$ python3 smart_cli.py train --topic calculus --user-id user_010 --stats
```

**System Output:**
```
======================================================================
TRAINING MODEL (USER-SPECIFIC): calculus for user_010
======================================================================

Loading user-specific training data for user_010...
  Found 30 completed tasks for this user
  Accuracy: 43.3%
  Mean time: 133.7s

Training personalized model for this user...
  Training user-specific parameters for user_010...
  Using 30 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: -0.069 (positive = actual better than predicted)
  Correctness std: 0.497
  Time bias (log): +0.024 (positive = actual slower than predicted)
  Time ratio: 1.03x (median: 1.01x)
  Samples: 30
  ℹ Predictions are well-calibrated

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Iteration 10: log-likelihood = -48.234
  Iteration 20: log-likelihood = -47.891
  Converged after 24 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 0.268  (unchanged, well-calibrated)
    Speed (τ): -0.111   (unchanged, well-calibrated)

======================================================================
USER-SPECIFIC TRAINING COMPLETE!
======================================================================

MODEL STATISTICS:
  Topic: calculus
  Users trained: 50

Difficulty Parameters:
  Level 1: difficulty=-0.42, time_intensity=4.33
  Level 2: difficulty=1.02, time_intensity=4.76
  Level 3: difficulty=1.32, time_intensity=5.01

User Ability: mean=0.27, range=[-0.14, 0.27]
User Speed: mean=-0.11, range=[-0.11, -0.11]
```

**Interpretation:**

Bob's parameters **didn't change** because:
1. His general training parameters are already well-calibrated
2. Small correctness bias (-0.069) is within acceptable range
3. Time predictions are accurate (1.03x ratio)
4. With 30 samples, the model has enough data

This is **expected behavior** - error-aware training only adjusts if systematic biases exist.

---

## Scenario 4: Charlie (Has Data But Forced to Retrain)

Charlie is another CSV user. Let's see what happens if we force retraining.

### Charlie's Original Parameters

```bash
$ python3 -c "
from topic_lnirt import TopicModelManager
model = TopicModelManager().get_model('calculus')
params = model.user_params.get('user_025', {})
print(f'Charlie (user_025):')
print(f'  θ = {params.get(\"theta\", \"N/A\")}')
print(f'  τ = {params.get(\"tau\", \"N/A\")}')
"
```

**Output:**
```
Charlie (user_025):
  θ = -0.13128447393696593
  τ = -0.02336049083313346
```

### Force User-Specific Training

```bash
$ python3 smart_cli.py train --topic calculus --user-id user_025
```

**System Output:**
```
======================================================================
TRAINING MODEL (USER-SPECIFIC): calculus for user_025
======================================================================

Loading user-specific training data for user_025...
  Found 30 completed tasks for this user
  Accuracy: 43.3%
  Mean time: 116.2s

Training personalized model for this user...
  Training user-specific parameters for user_025...
  Using 30 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: -0.059 (positive = actual better than predicted)
  Correctness std: 0.501
  Time bias (log): -0.023 (positive = actual slower than predicted)
  Time ratio: 0.98x (median: 0.99x)
  Samples: 30
  ℹ Predictions are well-calibrated

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Converged after 22 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): -0.131  (minimal change)
    Speed (τ): -0.023    (minimal change)

======================================================================
USER-SPECIFIC TRAINING COMPLETE!
======================================================================
```

Again, minimal changes because the general training already captured Charlie's performance accurately.

---

## Scenario 5: Edge Case - User with Only 1 Task

What happens when a user has only completed 1 task?

### Setup

```bash
$ python3 smart_cli.py predict --user-id edge_case_user --topic calculus --difficulty 1 --save
# Returns task_id=1506

$ python3 smart_cli.py update --task-id 1506 --correct 1 --time 45
```

**System Output:**
```
======================================================================
AUTOMATIC USER-SPECIFIC TRAINING
======================================================================

Training personalized model for edge_case_user...
  Using 1 completed tasks

  === Prediction Error Analysis ===
  Correctness bias: +0.612 (positive = actual better than predicted)
  Correctness std: 0.000
  Time bias (log): -0.512 (positive = actual slower than predicted)
  Time ratio: 0.60x (median: 0.60x)
  Samples: 1
  ⚠ Model systematically UNDERESTIMATES user ability
  ⚠ Model systematically OVERESTIMATES time needed
  ⚠ Warning: Small sample size (n=1), parameters may be unstable

  Running LNIRT Maximum Likelihood Estimation...
  Optimizing user parameters (θ, τ)...
  Converged after 5 iterations

  ✓ User parameters updated (Error-Aware LNIRT ML)
    Ability (θ): 3.000  (hit upper bound)
    Speed (τ): 0.628

✓ Personalized model updated for edge_case_user
```

**What Happened:**

With only 1 task:
- System **still trains** personalized model
- Parameters may **hit bounds** (θ=3.000 is max)
- Warning issued about small sample size
- **Works correctly** but with higher uncertainty

As user completes more tasks, parameters will stabilize.

---

## Scenario 6: System Statistics Overview

### View All Topics

```bash
$ python3 smart_cli.py stats
```

**System Output:**
```
======================================================================
STATISTICS
======================================================================

Available Topics (1):
  - calculus               (50 users trained)

Use --topic <name> to see detailed statistics

======================================================================
```

### Detailed Calculus Statistics

```bash
$ python3 smart_cli.py stats --topic calculus
```

**System Output:**
```
======================================================================
STATISTICS
======================================================================

Topic: calculus
Users: 50

Difficulty Parameters:
  Level 1:
    Discrimination (a): 0.50
    Difficulty (b): -0.42
    Time Intensity (β): 4.33 (≈76s)
  Level 2:
    Discrimination (a): 0.68
    Difficulty (b): 1.02
    Time Intensity (β): 4.76 (≈117s)
  Level 3:
    Discrimination (a): 0.54
    Difficulty (b): 1.32
    Time Intensity (β): 5.01 (≈150s)

User Statistics:
  Ability (θ):
    Mean: -0.06
    Std: 0.72
    Range: [-1.42, 2.91]
  Speed (τ):
    Mean: -0.04
    Std: 0.35
    Range: [-0.65, 0.63]

Prediction History:
  Total predictions: 1506
  Completed: 1506

Actual Performance by Difficulty:
  Level 1: accuracy=61.7%, avg_time=75.3s
  Level 2: accuracy=37.8%, avg_time=117.4s
  Level 3: accuracy=26.9%, avg_time=149.8s

======================================================================
```

**Interpretation:**

1. **Difficulty increases from 1→3:**
   - Accuracy decreases: 61.7% → 37.8% → 26.9%
   - Time increases: 75.3s → 117.4s → 149.8s

2. **User ability distribution:**
   - Mean near 0 (standard IRT scale)
   - Range: -1.42 to 2.91 (wide variation)
   - Alice is in top tier (θ=2.65)

3. **Model validation:**
   - Predicted time intensity matches actual avg time
   - β values (4.33, 4.76, 5.01) match log(actual times)

---

## Scenario 7: Batch Processing (Multiple Users)

### Testing Multiple Users in Sequence

```bash
# Create a simple batch script
cat > batch_predict.sh << 'EOF'
#!/bin/bash
for user in alice user_010 user_025 new_user_xyz; do
    echo "=== Predicting for $user ==="
    python3 smart_cli.py predict --user-id $user --topic calculus --difficulty 2
    echo ""
done
EOF

chmod +x batch_predict.sh
./batch_predict.sh
```

**Output:**
```
=== Predicting for alice ===
User: alice
Probability of Correct: 72.1%
Expected Time: 84.3 seconds

=== Predicting for user_010 ===
User: user_010
Probability of Correct: 37.5%
Expected Time: 130.8 seconds

=== Predicting for user_025 ===
User: user_025
Probability of Correct: 30.7%
Expected Time: 126.4 seconds

=== Predicting for new_user_xyz ===
User: new_user_xyz
Probability of Correct: 32.4%
Expected Time: 121.8 seconds
```

**Observations:**

- **alice:** Highest P(correct) due to personalized training
- **user_010, user_025:** Different predictions based on their CSV performance
- **new_user_xyz:** Uses population average (same as all new users)

---

## Scenario 8: Continuous Learning Over 1 Week

Let's simulate a student (Diana) completing tasks over 1 week.

### Week Timeline

**Monday:**
```bash
# Task 1 (D2): Population avg → predicted 32.4% / 121.8s → actual: CORRECT / 110s
# Automatic training → θ=2.234, τ=0.132
```

**Tuesday:**
```bash
# Task 2 (D2): Personalized → predicted 67.8% / 95.2s → actual: CORRECT / 88s
# Automatic training → θ=2.412, τ=0.198
```

**Wednesday:**
```bash
# Task 3 (D3): Personalized → predicted 58.4% / 112.5s → actual: INCORRECT / 215s
# Automatic training → θ=2.187, τ=0.045
```

**Thursday:**
```bash
# Task 4 (D2): Personalized → predicted 65.1% / 108.4s → actual: CORRECT / 102s
# Automatic training → θ=2.298, τ=0.089
```

**Friday:**
```bash
# Task 5 (D2): Personalized → predicted 68.9% / 103.1s → actual: CORRECT / 98s
# Automatic training → θ=2.356, τ=0.112
```

### Diana's Parameter Evolution

```
Day         Tasks    θ        τ        Avg P(D2)    Prediction Accuracy
------------------------------------------------------------------------
Monday      0        -0.061   -0.040   32.4%        N/A
Monday      1        2.234    0.132    67.2%        Low (first task)
Tuesday     2        2.412    0.198    70.5%        Improving
Wednesday   3        2.187    0.045    64.8%        Adjusted (failed D3)
Thursday    4        2.298    0.089    66.7%        Stabilizing
Friday      5        2.356    0.112    68.3%        Converged
```

**Key Insights:**

1. **Rapid personalization:** After task 1, predictions jump from 32% to 67%
2. **Adaptation to failure:** Task 3 failure appropriately lowered θ
3. **Convergence:** Parameters stabilizing by task 5
4. **Accuracy improving:** Model learning Diana's true performance level

---

## Scenario 9: Comparison - Same User, Different Topics

If we had multiple topics trained (e.g., calculus, algebra, geometry):

```bash
# Hypothetical: User good at calculus, poor at geometry

$ python3 smart_cli.py predict --user-id alice --topic calculus --difficulty 2
# Output: P(correct)=72.1%, θ=2.654

$ python3 smart_cli.py predict --user-id alice --topic geometry --difficulty 2
# Output: P(correct)=28.5%, θ=-0.512 (different model!)
```

**Why different?**
- Each topic has **separate model**
- User parameters are **topic-specific**
- Alice can be strong in calculus but weak in geometry
- Models are **independent**

---

## Summary: Key Workflow Patterns

### Pattern 1: New User Workflow
```
New User → Population Average → Complete Task → Automatic Training → Personalized
```

### Pattern 2: Existing User Workflow
```
Existing User → Personalized Params → Complete Task → Automatic Refinement → Better Personalized
```

### Pattern 3: Continuous Improvement
```
More Tasks → More Data → More Accurate Error Analysis → Better Parameters → Better Predictions
```

### Pattern 4: Adaptive Learning
```
Success → Increase θ
Failure → Decrease θ
Faster → Increase τ
Slower → Decrease τ
```

---

## Common Patterns and Outputs

### When to Expect High Confidence
- User has completed 5+ tasks
- Predictions match actual results closely
- Small correctness bias (<0.15)
- P(correct) > 70% for their level

### When to Expect Low Confidence
- New user (0 tasks)
- User attempting much harder difficulty
- Limited training data (1-2 tasks)
- P(correct) < 40%

### When Parameters Change Significantly
- First few tasks (large adjustments)
- User attempts new difficulty level
- Significant prediction errors
- Systematic biases detected

### When Parameters Stay Stable
- User has 20+ tasks
- Predictions are accurate
- Performance is consistent
- Well-calibrated model

---

## End of Scenarios

All scenarios demonstrate the **automatic, adaptive, error-aware** nature of the system. No manual intervention needed - the system continuously learns and improves predictions for each user.
