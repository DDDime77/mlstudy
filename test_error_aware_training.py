#!/usr/bin/env python3
"""
Comprehensive test suite for error-aware training functionality
Tests BOTH predicted and actual data usage in user-specific training
"""

import pandas as pd
import numpy as np
import sys
from topic_lnirt import TopicLNIRTModel
from predictions_db import PredictionsDB

print("=" * 80)
print("COMPREHENSIVE ERROR-AWARE TRAINING TEST SUITE")
print("=" * 80)

# Test counter
tests_passed = 0
tests_failed = 0
test_details = []

def test_result(test_name, passed, details=""):
    global tests_passed, tests_failed, test_details
    if passed:
        tests_passed += 1
        status = "✓ PASS"
    else:
        tests_failed += 1
        status = "✗ FAIL"

    result_line = f"{status}: {test_name}"
    print(result_line)
    if details:
        print(f"       {details}")
    test_details.append((test_name, passed, details))

# ============================================================================
# TEST 1: Error Analysis Method Exists and Works
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Error Analysis Method")
print("=" * 80)

try:
    model = TopicLNIRTModel('test_topic')

    # Create sample data with predictions
    test_data = pd.DataFrame({
        'difficulty': [1, 1, 2, 2, 3, 3],
        'correct': [1, 1, 0, 1, 0, 0],
        'response_time': [30, 35, 80, 75, 150, 160],
        'predicted_correct': [0.8, 0.75, 0.4, 0.45, 0.2, 0.25],
        'predicted_time': [28, 32, 85, 78, 145, 155]
    })

    # Train basic model first
    train_data = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3', 'u1', 'u2', 'u3'],
        'difficulty': [1, 1, 2, 2, 3, 3],
        'correct': [1, 1, 0, 1, 0, 0],
        'response_time': [30, 35, 80, 75, 150, 160]
    })
    model.fit(train_data, verbose=False)

    # Test error analysis
    error_stats = model._analyze_prediction_errors(test_data, verbose=True)

    if error_stats is not None:
        required_keys = ['correctness_bias', 'time_bias_log', 'n_samples']
        has_all_keys = all(key in error_stats for key in required_keys)
        test_result("Error analysis returns valid statistics", has_all_keys,
                   f"Keys present: {list(error_stats.keys())}")
    else:
        test_result("Error analysis returns valid statistics", False,
                   "Returned None instead of statistics dict")

except Exception as e:
    test_result("Error analysis method execution", False, f"Exception: {str(e)}")

# ============================================================================
# TEST 2: User-Specific Training Uses Predicted Data
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: User-Specific Training with Predicted Data")
print("=" * 80)

try:
    model = TopicLNIRTModel('test_topic_2')

    # Create training data
    train_data = pd.DataFrame({
        'user_id': ['u1'] * 30,
        'difficulty': [1]*10 + [2]*10 + [3]*10,
        'correct': [1]*8 + [0]*2 + [1]*6 + [0]*4 + [1]*3 + [0]*7,
        'response_time': [30 + np.random.randn()*5 for _ in range(10)] +
                        [80 + np.random.randn()*10 for _ in range(10)] +
                        [150 + np.random.randn()*15 for _ in range(10)]
    })

    model.fit(train_data, verbose=False)

    # Get initial parameters
    initial_theta = model.user_params['u1']['theta']
    initial_tau = model.user_params['u1']['tau']

    print(f"\nInitial parameters: θ={initial_theta:.3f}, τ={initial_tau:.3f}")

    # Create user-specific data WITH systematic bias
    # Model predicts poorly: underestimates ability, overestimates time
    user_data = pd.DataFrame({
        'difficulty': [1]*10 + [2]*10 + [3]*10,
        'correct': [1]*9 + [0]*1 + [1]*7 + [0]*3 + [1]*5 + [0]*5,  # Better than predicted
        'response_time': [25]*10 + [70]*10 + [130]*10,  # Faster than predicted
        'predicted_correct': [0.6]*10 + [0.4]*10 + [0.2]*10,  # Underestimated
        'predicted_time': [35]*10 + [90]*10 + [160]*10  # Overestimated
    })

    # Train with error-aware method
    print("\nTraining with error-aware method...")
    model.fit_user_specific(user_data, 'u1', verbose=True)

    # Get new parameters
    new_theta = model.user_params['u1']['theta']
    new_tau = model.user_params['u1']['tau']

    print(f"\nUpdated parameters: θ={new_theta:.3f}, τ={new_tau:.3f}")
    print(f"Changes: Δθ={new_theta-initial_theta:+.3f}, Δτ={new_tau-initial_tau:+.3f}")

    # Verify theta increased (user is better than predicted)
    theta_increased = new_theta > initial_theta
    test_result("Theta increases when user performs better than predicted",
               theta_increased,
               f"Δθ={new_theta-initial_theta:+.3f}")

    # Verify tau increased (user is faster than predicted)
    tau_increased = new_tau > initial_tau
    test_result("Tau increases when user is faster than predicted",
               tau_increased,
               f"Δτ={new_tau-initial_tau:+.3f}")

except Exception as e:
    test_result("User-specific training with predicted data", False,
               f"Exception: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: Database Integration - Predicted Data Storage
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Database Storage of Predicted and Actual Data")
print("=" * 80)

try:
    import os
    test_db_path = "test_predictions.db"

    # Remove old test db
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = PredictionsDB(test_db_path)

    # Save prediction with predicted values
    task_id = db.add_prediction(
        user_id='test_user',
        topic='calculus',
        difficulty=2,
        predicted_correct=0.65,
        predicted_time=120.5
    )

    test_result("Save prediction to database", task_id > 0,
               f"Task ID: {task_id}")

    # Update with actual values
    db.update_prediction(
        task_id=task_id,
        actual_correct=1,
        actual_time=105.3
    )

    # Query user training data
    user_data = db.get_user_training_data('test_user', 'calculus')

    has_predicted = 'predicted_correct' in user_data.columns
    has_actual = 'correct' in user_data.columns
    test_result("Database returns both predicted and actual data",
               has_predicted and has_actual,
               f"Columns: {list(user_data.columns)}")

    if len(user_data) > 0:
        row = user_data.iloc[0]
        values_match = (row['correct'] == 1 and
                       abs(row['response_time'] - 105.3) < 0.1 and
                       abs(row['predicted_correct'] - 0.65) < 0.01 and
                       abs(row['predicted_time'] - 120.5) < 0.1)
        test_result("Database stores correct values",
                   values_match,
                   f"Predicted: {row['predicted_correct']:.2f}, {row['predicted_time']:.1f}s | "
                   f"Actual: {row['correct']}, {row['response_time']:.1f}s")

    # Cleanup
    os.remove(test_db_path)

except Exception as e:
    test_result("Database integration", False, f"Exception: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Full Workflow with Error Awareness
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Complete Workflow with Error-Aware Training")
print("=" * 80)

try:
    test_db_path = "test_workflow.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    db = PredictionsDB(test_db_path)
    model = TopicLNIRTModel('calculus')

    # Step 1: Train general model
    general_data = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3'] * 20,
        'difficulty': ([1]*20 + [2]*20 + [3]*20),
        'correct': ([1]*15 + [0]*5 + [1]*12 + [0]*8 + [1]*8 + [0]*12),
        'response_time': [30 + np.random.randn()*5 for _ in range(20)] +
                        [80 + np.random.randn()*10 for _ in range(20)] +
                        [150 + np.random.randn()*15 for _ in range(20)]
    })

    model.fit(general_data, verbose=False)
    print("✓ General model trained")

    # Step 2: Make predictions for a user and save them
    user_id = 'student_xyz'
    predictions_made = []

    for diff in [1, 1, 1, 2, 2, 3]:
        p_correct, p_time = model.predict(user_id, diff)
        task_id = db.add_prediction(user_id, 'calculus', diff, p_correct, p_time)
        predictions_made.append({
            'task_id': task_id,
            'difficulty': diff,
            'predicted_correct': p_correct,
            'predicted_time': p_time
        })

    print(f"✓ Made {len(predictions_made)} predictions")

    # Step 3: Simulate user completing tasks (performs better than predicted)
    actual_results = [
        (1, 25),  # correct, faster
        (1, 28),  # correct, faster
        (1, 30),  # correct, faster
        (1, 75),  # correct, faster
        (0, 95),  # incorrect, slower
        (1, 140), # correct, faster
    ]

    for pred, (actual_correct, actual_time) in zip(predictions_made, actual_results):
        db.update_prediction(pred['task_id'], actual_correct, actual_time)

    print("✓ Updated with actual results")

    # Step 4: Train user-specific model
    user_data = db.get_user_training_data(user_id, 'calculus')

    print(f"\nUser data shape: {user_data.shape}")
    print(f"Columns: {list(user_data.columns)}")

    # Get parameters before training
    if user_id in model.user_params:
        theta_before = model.user_params[user_id]['theta']
        tau_before = model.user_params[user_id]['tau']
    else:
        theta_before = 0.0
        tau_before = 0.0

    model.fit_user_specific(user_data, user_id, verbose=True)

    theta_after = model.user_params[user_id]['theta']
    tau_after = model.user_params[user_id]['tau']

    print(f"\nParameter changes:")
    print(f"  Theta: {theta_before:.3f} → {theta_after:.3f} (Δ={theta_after-theta_before:+.3f})")
    print(f"  Tau: {tau_before:.3f} → {tau_after:.3f} (Δ={tau_after-tau_before:+.3f})")

    # Step 5: Make new prediction and verify it's different
    p_correct_new, p_time_new = model.predict(user_id, 2)

    # Find original prediction for difficulty 2
    original_pred = [p for p in predictions_made if p['difficulty'] == 2][0]

    prediction_changed = (abs(p_correct_new - original_pred['predicted_correct']) > 0.01 or
                         abs(p_time_new - original_pred['predicted_time']) > 1.0)

    test_result("Predictions change after error-aware training",
               prediction_changed,
               f"Original: {original_pred['predicted_correct']:.2%}, {original_pred['predicted_time']:.1f}s | "
               f"New: {p_correct_new:.2%}, {p_time_new:.1f}s")

    # Cleanup
    os.remove(test_db_path)

except Exception as e:
    test_result("Full workflow", False, f"Exception: {str(e)}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: Edge Cases
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Edge Cases")
print("=" * 80)

# Test 5.1: No predicted data (missing columns)
try:
    model = TopicLNIRTModel('edge_test')
    train_data = pd.DataFrame({
        'user_id': ['u1'] * 5,
        'difficulty': [1, 2, 3, 1, 2],
        'correct': [1, 0, 0, 1, 1],
        'response_time': [30, 80, 150, 35, 75]
    })
    model.fit(train_data, verbose=False)

    # Data without predicted columns
    user_data_no_pred = pd.DataFrame({
        'difficulty': [1, 2, 3],
        'correct': [1, 1, 0],
        'response_time': [30, 75, 145]
    })

    model.fit_user_specific(user_data_no_pred, 'u1', verbose=False)
    test_result("Handles missing predicted columns gracefully", True,
               "No exception raised")

except Exception as e:
    test_result("Handles missing predicted columns gracefully", False,
               f"Exception: {str(e)}")

# Test 5.2: Single data point
try:
    model = TopicLNIRTModel('edge_test_2')
    train_data = pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'difficulty': [1, 2],
        'correct': [1, 0],
        'response_time': [30, 80]
    })
    model.fit(train_data, verbose=False)

    single_point = pd.DataFrame({
        'difficulty': [1],
        'correct': [1],
        'response_time': [30],
        'predicted_correct': [0.5],
        'predicted_time': [35]
    })

    model.fit_user_specific(single_point, 'u1', verbose=False)
    test_result("Handles single data point", True,
               "No exception raised")

except Exception as e:
    test_result("Handles single data point", False,
               f"Exception: {str(e)}")

# Test 5.3: Extreme values
try:
    model = TopicLNIRTModel('edge_test_3')
    train_data = pd.DataFrame({
        'user_id': ['u1'] * 6,
        'difficulty': [1, 1, 2, 2, 3, 3],
        'correct': [1, 1, 1, 0, 0, 0],
        'response_time': [30, 35, 80, 85, 150, 160]
    })
    model.fit(train_data, verbose=False)

    # All correct, very fast
    extreme_data = pd.DataFrame({
        'difficulty': [1, 2, 3],
        'correct': [1, 1, 1],
        'response_time': [10, 20, 30],
        'predicted_correct': [0.5, 0.5, 0.5],
        'predicted_time': [50, 100, 200]
    })

    model.fit_user_specific(extreme_data, 'u1', verbose=False)

    theta = model.user_params['u1']['theta']
    tau = model.user_params['u1']['tau']

    # Should have high ability and high speed
    params_reasonable = -3.0 <= theta <= 3.0 and -3.0 <= tau <= 3.0
    test_result("Handles extreme values with bounds", params_reasonable,
               f"θ={theta:.3f}, τ={tau:.3f}")

except Exception as e:
    test_result("Handles extreme values", False,
               f"Exception: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"\nTotal Tests: {tests_passed + tests_failed}")
print(f"Passed: {tests_passed} ✓")
print(f"Failed: {tests_failed} ✗")

if tests_failed > 0:
    print("\nFailed Tests:")
    for name, passed, details in test_details:
        if not passed:
            print(f"  ✗ {name}")
            if details:
                print(f"    {details}")
    sys.exit(1)
else:
    print("\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
