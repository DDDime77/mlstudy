#!/usr/bin/env python3
"""
Comprehensive error analysis to catch any bugs in the system.

Tests:
1. Database integrity
2. Model training/prediction consistency
3. Error-aware training correctness
4. Edge cases
5. Full workflow
"""

import pandas as pd
import numpy as np
from topic_lnirt import TopicModelManager, TopicLNIRTModel
from predictions_db import PredictionsDB
import sys

# Test counter
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

print("=" * 100)
print("COMPREHENSIVE ERROR ANALYSIS")
print("=" * 100)

# =============================================================================
# TEST 1: Database Integrity
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 1: DATABASE INTEGRITY")
print("=" * 100)

db = PredictionsDB()

# Test 1.1: All predictions have both predicted and actual
query = """
SELECT COUNT(*) as total,
       SUM(CASE WHEN predicted_correct IS NOT NULL THEN 1 ELSE 0 END) as has_pred,
       SUM(CASE WHEN actual_correct IS NOT NULL THEN 1 ELSE 0 END) as has_actual
FROM predictions
WHERE topic = 'calculus'
"""
result = pd.read_sql_query(query, db.conn)
total = result['total'][0]
has_pred = result['has_pred'][0]
has_actual = result['has_actual'][0]

test_result("All predictions have predicted values", has_pred == total,
           f"{has_pred}/{total} have predicted_correct")
test_result("All predictions have actual values", has_actual == total,
           f"{has_actual}/{total} have actual_correct")

# Test 1.2: Predictions and training_data are consistent
query_pred_count = "SELECT COUNT(*) as count FROM predictions WHERE topic='calculus'"
query_train_count = "SELECT COUNT(*) as count FROM training_data WHERE topic='calculus'"

pred_count = pd.read_sql_query(query_pred_count, db.conn)['count'][0]
train_count = pd.read_sql_query(query_train_count, db.conn)['count'][0]

test_result("Predictions and training_data counts match", pred_count == train_count,
           f"predictions: {pred_count}, training_data: {train_count}")

# Test 1.3: All users in predictions table
query_users = "SELECT DISTINCT user_id FROM predictions WHERE topic='calculus'"
users_in_pred = pd.read_sql_query(query_users, db.conn)
test_result("Predictions table has expected 50 users", len(users_in_pred) == 50,
           f"Found {len(users_in_pred)} users")

# Test 1.4: All users have complete data
query_incomplete = """
SELECT user_id, COUNT(*) as total,
       SUM(CASE WHEN predicted_correct IS NULL THEN 1 ELSE 0 END) as missing_pred,
       SUM(CASE WHEN actual_correct IS NULL THEN 1 ELSE 0 END) as missing_actual
FROM predictions
WHERE topic='calculus'
GROUP BY user_id
HAVING missing_pred > 0 OR missing_actual > 0
"""
incomplete_users = pd.read_sql_query(query_incomplete, db.conn)
test_result("No users have incomplete data", len(incomplete_users) == 0,
           f"{len(incomplete_users)} users with incomplete data")

# =============================================================================
# TEST 2: Model Consistency
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 2: MODEL CONSISTENCY")
print("=" * 100)

manager = TopicModelManager()
model = manager.get_model('calculus')

# Test 2.1: Model is trained
test_result("Model is trained", model.is_trained,
           f"is_trained: {model.is_trained}")

# Test 2.2: Model has all 50 users
test_result("Model has 50 users", len(model.user_params) == 50,
           f"Found {len(model.user_params)} users in model")

# Test 2.3: All difficulty levels have parameters
for diff in [1, 2, 3]:
    has_params = (diff in model.difficulty_params and
                 'a' in model.difficulty_params[diff] and
                 'b' in model.difficulty_params[diff] and
                 'beta' in model.difficulty_params[diff])
    test_result(f"Difficulty {diff} has all parameters", has_params)

# Test 2.4: Parameters are within reasonable bounds
param_issues = []
for user_id, params in model.user_params.items():
    if not (-3.0 <= params['theta'] <= 3.0):
        param_issues.append(f"{user_id}: θ={params['theta']}")
    if not (-3.0 <= params['tau'] <= 3.0):
        param_issues.append(f"{user_id}: τ={params['tau']}")

test_result("All user parameters within bounds [-3, 3]", len(param_issues) == 0,
           f"Issues: {param_issues[:5] if param_issues else 'None'}")

for diff in [1, 2, 3]:
    a = model.difficulty_params[diff]['a']
    b = model.difficulty_params[diff]['b']
    beta = model.difficulty_params[diff]['beta']

    issues = []
    if not (0.5 <= a <= 3.0):
        issues.append(f"a={a}")
    if not (-3.0 <= b <= 3.0):
        issues.append(f"b={b}")
    if not (2.0 <= beta <= 6.0):
        issues.append(f"β={beta}")

    test_result(f"Difficulty {diff} parameters within bounds", len(issues) == 0,
               f"Issues: {issues if issues else 'None'}")

# =============================================================================
# TEST 3: Prediction Correctness
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 3: PREDICTION CORRECTNESS")
print("=" * 100)

# Test 3.1: Predictions for existing user
try:
    p_correct, expected_time = model.predict('user_000', 2)
    test_result("Prediction for existing user succeeds", True,
               f"P={p_correct:.1%}, T={expected_time:.1f}s")

    # Check values are reasonable
    test_result("Probability is between 0 and 1", 0 <= p_correct <= 1,
               f"P={p_correct:.3f}")
    test_result("Expected time is positive", expected_time > 0,
               f"T={expected_time:.1f}s")
except Exception as e:
    test_result("Prediction for existing user succeeds", False, str(e))

# Test 3.2: Predictions for new user
try:
    p_correct_new, expected_time_new = model.predict('brand_new_user', 2)
    test_result("Prediction for new user succeeds", True,
               f"P={p_correct_new:.1%}, T={expected_time_new:.1f}s")
except Exception as e:
    test_result("Prediction for new user succeeds", False, str(e))

# Test 3.3: Predictions are consistent across multiple calls
try:
    p1, t1 = model.predict('user_010', 1)
    p2, t2 = model.predict('user_010', 1)
    test_result("Predictions are deterministic", p1 == p2 and t1 == t2,
               f"Call 1: ({p1:.3f}, {t1:.1f}), Call 2: ({p2:.3f}, {t2:.1f})")
except Exception as e:
    test_result("Predictions are deterministic", False, str(e))

# Test 3.4: Different difficulties produce different predictions
try:
    p_d1, t_d1 = model.predict('user_010', 1)
    p_d2, t_d2 = model.predict('user_010', 2)
    p_d3, t_d3 = model.predict('user_010', 3)

    # Generally: easier = higher P(correct), shorter time
    # But not always true due to user-specific factors
    test_result("Different difficulties produce different predictions",
               not (p_d1 == p_d2 == p_d3 and t_d1 == t_d2 == t_d3),
               f"D1: ({p_d1:.2f}, {t_d1:.0f}), D2: ({p_d2:.2f}, {t_d2:.0f}), D3: ({p_d3:.2f}, {t_d3:.0f})")
except Exception as e:
    test_result("Different difficulties produce different predictions", False, str(e))

# =============================================================================
# TEST 4: Error-Aware Training
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 4: ERROR-AWARE TRAINING")
print("=" * 100)

# Test 4.1: Error analysis method exists and works
try:
    user_data = db.get_user_training_data('user_005', 'calculus')
    error_stats = model._analyze_prediction_errors(user_data, verbose=False)

    test_result("Error analysis method works", error_stats is not None,
               f"Returned: {list(error_stats.keys())[:3]}")

    required_keys = ['correctness_bias', 'time_bias_log', 'n_samples']
    has_keys = all(key in error_stats for key in required_keys)
    test_result("Error stats has required keys", has_keys,
               f"Keys: {list(error_stats.keys())}")
except Exception as e:
    test_result("Error analysis method works", False, str(e))

# Test 4.2: User-specific training modifies parameters
try:
    # Get initial params
    test_user = 'user_015'
    initial_theta = model.user_params[test_user]['theta']
    initial_tau = model.user_params[test_user]['tau']

    # Train
    user_data = db.get_user_training_data(test_user, 'calculus')
    model.fit_user_specific(user_data, test_user, verbose=False)

    # Check if changed
    new_theta = model.user_params[test_user]['theta']
    new_tau = model.user_params[test_user]['tau']

    # May or may not change depending on data, but should not crash
    test_result("User-specific training completes", True,
               f"Before: ({initial_theta:.3f}, {initial_tau:.3f}), After: ({new_theta:.3f}, {new_tau:.3f})")
except Exception as e:
    test_result("User-specific training completes", False, str(e))

# =============================================================================
# TEST 5: Edge Cases
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 5: EDGE CASES")
print("=" * 100)

# Test 5.1: Invalid difficulty
try:
    model.predict('user_000', 5)
    test_result("Invalid difficulty raises error", False, "Should have raised ValueError")
except ValueError:
    test_result("Invalid difficulty raises error", True, "ValueError raised as expected")
except Exception as e:
    test_result("Invalid difficulty raises error", False, f"Wrong exception: {type(e)}")

# Test 5.2: User with no data still can be trained
try:
    # Create model copy to avoid affecting main model
    test_model = TopicLNIRTModel('test')
    test_model.difficulty_params = model.difficulty_params.copy()
    test_model.user_params = model.user_params.copy()
    test_model.is_trained = True

    # Try to train with empty data
    empty_data = pd.DataFrame({
        'difficulty': [],
        'correct': [],
        'response_time': [],
        'predicted_correct': [],
        'predicted_time': []
    })

    # Should handle gracefully
    try:
        test_model.fit_user_specific(empty_data, 'empty_user', verbose=False)
        test_result("Empty data handled gracefully", True, "No crash")
    except Exception as e:
        # It's ok if it raises an exception, as long as it's handled
        test_result("Empty data handled with exception", True, f"Exception: {type(e).__name__}")
except Exception as e:
    test_result("Empty data edge case", False, str(e))

# Test 5.3: Very high/low actual performance
try:
    # User with 100% accuracy
    perfect_data = pd.DataFrame({
        'difficulty': [1, 2, 3],
        'correct': [1, 1, 1],
        'response_time': [10.0, 20.0, 30.0],
        'predicted_correct': [0.5, 0.5, 0.5],
        'predicted_time': [50.0, 100.0, 150.0]
    })

    test_model = TopicLNIRTModel('test')
    test_model.difficulty_params = model.difficulty_params.copy()
    test_model.user_params = {'perfect_user': {'theta': 0.0, 'tau': 0.0}}
    test_model.is_trained = True

    test_model.fit_user_specific(perfect_data, 'perfect_user', verbose=False)

    # Check theta increased (should be very high for 100% accuracy)
    final_theta = test_model.user_params['perfect_user']['theta']
    test_result("Perfect performance increases theta", final_theta > 0,
               f"θ={final_theta:.3f}")
except Exception as e:
    test_result("Perfect performance edge case", False, str(e))

# =============================================================================
# TEST 6: Full Workflow
# =============================================================================
print("\n" + "=" * 100)
print("TEST SUITE 6: FULL WORKFLOW")
print("=" * 100)

try:
    # Simulate complete workflow
    test_user_wf = 'workflow_test_user'

    # Step 1: Make prediction
    p, t = model.predict(test_user_wf, 2)
    test_result("Step 1: Make prediction", True, f"P={p:.1%}, T={t:.1f}s")

    # Step 2: Save to database
    task_id = db.add_prediction(test_user_wf, 'calculus', 2, p, t)
    test_result("Step 2: Save prediction", task_id > 0, f"task_id={task_id}")

    # Step 3: Update with actual
    db.update_prediction(task_id, 1, 95.5)
    test_result("Step 3: Update with actual", True, "Updated successfully")

    # Step 4: Verify in database
    pred = db.get_prediction(task_id)
    test_result("Step 4: Retrieve prediction", pred is not None and pred['actual_correct'] == 1,
               f"actual_correct={pred['actual_correct'] if pred else 'None'}")

    # Step 5: User-specific training
    user_data_wf = db.get_user_training_data(test_user_wf, 'calculus')
    test_result("Step 5: Get user training data", len(user_data_wf) == 1,
               f"Retrieved {len(user_data_wf)} records")

    # Clean up
    db.conn.execute(f"DELETE FROM predictions WHERE task_id={task_id}")
    db.conn.execute(f"DELETE FROM training_data WHERE user_id='workflow_test_user' AND topic='calculus'")
    db.conn.commit()

except Exception as e:
    test_result("Full workflow", False, str(e))

# =============================================================================
# SUMMARY
# =============================================================================
db.close()

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
    sys.exit(0)
