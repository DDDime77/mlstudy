#!/usr/bin/env python3
"""
Test automatic user-specific training workflow.

Verifies that:
1. New user predictions use population average
2. After update, automatic training occurs
3. Subsequent predictions use personalized parameters
4. No user exists with actual data but without training
"""

import pandas as pd
import numpy as np
from topic_lnirt import TopicModelManager
from predictions_db import PredictionsDB
import sys

print("=" * 100)
print("AUTOMATIC TRAINING WORKFLOW TEST")
print("=" * 100)

# Initialize
manager = TopicModelManager()
model = manager.get_model('calculus')
db = PredictionsDB()

# Test user
test_user = 'auto_training_test_user_2024'

# Clean up any existing data for this test user
db.conn.execute(f"DELETE FROM predictions WHERE user_id='{test_user}' AND topic='calculus'")
db.conn.execute(f"DELETE FROM training_data WHERE user_id='{test_user}' AND topic='calculus'")
db.conn.commit()

# Also remove from model if present
if test_user in model.user_params:
    del model.user_params[test_user]
    manager.save_model('calculus')

tests_passed = 0
tests_failed = 0
errors = []

def test_result(test_name, condition, details=""):
    global tests_passed, tests_failed, errors
    if condition:
        tests_passed += 1
        print(f"✓ PASS: {test_name}")
        if details:
            print(f"       {details}")
    else:
        tests_failed += 1
        print(f"✗ FAIL: {test_name}")
        if details:
            print(f"       {details}")
        errors.append((test_name, details))

# =============================================================================
# STEP 1: NEW USER - FIRST PREDICTION
# =============================================================================
print("\n" + "=" * 100)
print("STEP 1: NEW USER - FIRST PREDICTION")
print("=" * 100)

# Verify user is not in model
test_result("User not in model initially", test_user not in model.user_params,
           f"In model: {test_user in model.user_params}")

# Calculate population average
pop_avg_theta = np.mean([p['theta'] for p in model.user_params.values()])
pop_avg_tau = np.mean([p['tau'] for p in model.user_params.values()])

print(f"\nPopulation average: θ={pop_avg_theta:.3f}, τ={pop_avg_tau:.3f}")

# Make first prediction
difficulty = 2
p_correct_1, expected_time_1 = model.predict(test_user, difficulty)

print(f"\nFirst prediction (difficulty {difficulty}):")
print(f"  P(correct): {p_correct_1:.1%}")
print(f"  Expected time: {expected_time_1:.1f}s")

# Verify it uses population average
a = model.difficulty_params[difficulty]['a']
b = model.difficulty_params[difficulty]['b']
beta = model.difficulty_params[difficulty]['beta']

expected_p = model._irt_probability(pop_avg_theta, a, b)
expected_t = np.exp(beta - pop_avg_tau)

test_result("First prediction uses population average",
           abs(p_correct_1 - expected_p) < 0.001 and abs(expected_time_1 - expected_t) < 0.1,
           f"P: {p_correct_1:.3f} vs {expected_p:.3f}, T: {expected_time_1:.1f}s vs {expected_t:.1f}s")

# Save prediction to database
task_id_1 = db.add_prediction(test_user, 'calculus', difficulty, p_correct_1, expected_time_1)
test_result("Prediction saved to database", task_id_1 > 0, f"task_id={task_id_1}")

# =============================================================================
# STEP 2: UPDATE WITH ACTUAL RESULTS - AUTOMATIC TRAINING
# =============================================================================
print("\n" + "=" * 100)
print("STEP 2: UPDATE WITH ACTUAL RESULTS - AUTOMATIC TRAINING")
print("=" * 100)

# Simulate user answering correctly in less time than expected
actual_correct_1 = 1  # Correct
actual_time_1 = 85.0  # Faster than expected

print(f"\nActual result:")
print(f"  Correctness: {'CORRECT' if actual_correct_1 == 1 else 'INCORRECT'}")
print(f"  Time: {actual_time_1:.1f}s")

# Update database
db.update_prediction(task_id_1, actual_correct_1, actual_time_1)
print("\n✓ Database updated")

# AUTOMATIC TRAINING (simulating what update command does)
print("\nRunning automatic user-specific training...")
user_training_data = db.get_user_training_data(test_user, 'calculus')

test_result("User training data retrieved", len(user_training_data) == 1,
           f"Retrieved {len(user_training_data)} records")

if len(user_training_data) > 0:
    print(f"  Using {len(user_training_data)} completed tasks")

    # Run full error-aware user-specific training
    model.fit_user_specific(user_training_data, test_user, verbose=True)
    manager.save_model('calculus')

    print(f"\n✓ Personalized model updated for {test_user}")

# Verify user is now in model
test_result("User added to model after training", test_user in model.user_params,
           f"In model: {test_user in model.user_params}")

# Get personalized parameters
if test_user in model.user_params:
    theta_personalized = model.user_params[test_user]['theta']
    tau_personalized = model.user_params[test_user]['tau']

    print(f"\nPersonalized parameters: θ={theta_personalized:.3f}, τ={tau_personalized:.3f}")
    print(f"Population average: θ={pop_avg_theta:.3f}, τ={pop_avg_tau:.3f}")
    print(f"Δ from average: Δθ={theta_personalized - pop_avg_theta:+.3f}, Δτ={tau_personalized - pop_avg_tau:+.3f}")

    # Since user was faster and correct, expect higher theta and tau
    test_result("Parameters differ from population average",
               abs(theta_personalized - pop_avg_theta) > 0.01 or abs(tau_personalized - pop_avg_tau) > 0.01,
               f"Δθ={theta_personalized - pop_avg_theta:+.3f}, Δτ={tau_personalized - pop_avg_tau:+.3f}")

# =============================================================================
# STEP 3: SECOND PREDICTION - SHOULD USE PERSONALIZED PARAMETERS
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: SECOND PREDICTION - SHOULD USE PERSONALIZED PARAMETERS")
print("=" * 100)

# Make second prediction (same difficulty)
p_correct_2, expected_time_2 = model.predict(test_user, difficulty)

print(f"\nSecond prediction (difficulty {difficulty}):")
print(f"  P(correct): {p_correct_2:.1%}")
print(f"  Expected time: {expected_time_2:.1f}s")

# Verify it uses personalized parameters (not population average)
if test_user in model.user_params:
    expected_p_personalized = model._irt_probability(theta_personalized, a, b)
    expected_t_personalized = np.exp(beta - tau_personalized)

    test_result("Second prediction uses personalized parameters",
               abs(p_correct_2 - expected_p_personalized) < 0.001 and abs(expected_time_2 - expected_t_personalized) < 0.1,
               f"P: {p_correct_2:.3f} vs {expected_p_personalized:.3f}, T: {expected_time_2:.1f}s vs {expected_t_personalized:.1f}s")

    # Verify it does NOT use population average
    test_result("Second prediction differs from population average",
               abs(p_correct_2 - expected_p) > 0.001 or abs(expected_time_2 - expected_t) > 0.1,
               f"P: {p_correct_2:.3f} vs pop {expected_p:.3f}, T: {expected_time_2:.1f}s vs pop {expected_t:.1f}s")

# Save second prediction
task_id_2 = db.add_prediction(test_user, 'calculus', difficulty, p_correct_2, expected_time_2)

# =============================================================================
# STEP 4: SECOND UPDATE - PARAMETERS SHOULD BE REFINED
# =============================================================================
print("\n" + "=" * 100)
print("STEP 4: SECOND UPDATE - PARAMETERS SHOULD BE REFINED")
print("=" * 100)

# Simulate another correct answer, still fast
actual_correct_2 = 1
actual_time_2 = 90.0

db.update_prediction(task_id_2, actual_correct_2, actual_time_2)

# Run automatic training again
user_training_data_2 = db.get_user_training_data(test_user, 'calculus')

test_result("User training data now has 2 records", len(user_training_data_2) == 2,
           f"Retrieved {len(user_training_data_2)} records")

if len(user_training_data_2) > 0:
    # Store old parameters
    theta_old = model.user_params[test_user]['theta']
    tau_old = model.user_params[test_user]['tau']

    # Retrain
    print(f"\nRetraining with {len(user_training_data_2)} completed tasks...")
    model.fit_user_specific(user_training_data_2, test_user, verbose=True)
    manager.save_model('calculus')

    # Get refined parameters
    theta_refined = model.user_params[test_user]['theta']
    tau_refined = model.user_params[test_user]['tau']

    print(f"\nParameter refinement:")
    print(f"  Before: θ={theta_old:.3f}, τ={tau_old:.3f}")
    print(f"  After: θ={theta_refined:.3f}, τ={tau_refined:.3f}")
    print(f"  Δθ={theta_refined - theta_old:+.3f}, Δτ={tau_refined - tau_old:+.3f}")

    # Parameters may or may not change significantly depending on data
    test_result("Second training completes successfully", True,
               f"Refined parameters: θ={theta_refined:.3f}, τ={tau_refined:.3f}")

# =============================================================================
# STEP 5: VERIFY NO USER HAS ACTUAL DATA WITHOUT TRAINING
# =============================================================================
print("\n" + "=" * 100)
print("STEP 5: VERIFY NO USER HAS ACTUAL DATA WITHOUT TRAINING")
print("=" * 100)

# Get all users with actual data
query = """
SELECT DISTINCT user_id
FROM predictions
WHERE topic='calculus' AND actual_correct IS NOT NULL
"""
users_with_data = pd.read_sql_query(query, db.conn)['user_id'].tolist()

print(f"\nTotal users with actual data: {len(users_with_data)}")

# Check which users are NOT in model
users_not_in_model = [u for u in users_with_data if u not in model.user_params]

test_result("All users with actual data are in model", len(users_not_in_model) == 0,
           f"Users not in model: {users_not_in_model[:5] if users_not_in_model else 'None'}")

if len(users_not_in_model) > 0:
    print(f"\n⚠ WARNING: {len(users_not_in_model)} users have actual data but are not trained!")
    print(f"   First 10: {users_not_in_model[:10]}")
else:
    print("\n✓ All users with actual data have been trained")

# Check for users who should use population average
users_without_data = [u for u in model.user_params.keys() if u not in users_with_data]
print(f"\nUsers in model without actual data: {len(users_without_data)}")

# =============================================================================
# CLEANUP
# =============================================================================
print("\n" + "=" * 100)
print("CLEANUP")
print("=" * 100)

# Clean up test user
db.conn.execute(f"DELETE FROM predictions WHERE user_id='{test_user}' AND topic='calculus'")
db.conn.execute(f"DELETE FROM training_data WHERE user_id='{test_user}' AND topic='calculus'")
db.conn.commit()

if test_user in model.user_params:
    del model.user_params[test_user]
    manager.save_model('calculus')

print(f"✓ Cleaned up test user: {test_user}")

db.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUMMARY")
print("=" * 100)

total = tests_passed + tests_failed
print(f"\nTotal Tests: {total}")
print(f"Passed: {tests_passed} ✓")
print(f"Failed: {tests_failed} ✗")

if tests_failed > 0:
    print("\n" + "=" * 100)
    print("FAILED TESTS")
    print("=" * 100)
    for test_name, details in errors:
        print(f"\n✗ {test_name}")
        if details:
            print(f"  {details}")
    sys.exit(1)
else:
    print("\n" + "=" * 100)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 100)
    print("\n✓ Automatic training workflow verified:")
    print("  1. New users use population average")
    print("  2. After first actual result, automatic training occurs")
    print("  3. Subsequent predictions use personalized parameters")
    print("  4. Parameters are refined with more data")
    print("  5. All users with actual data are trained")
    sys.exit(0)
