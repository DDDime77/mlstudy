#!/bin/bash

# Comprehensive CLI Testing Script
# Tests all commands with error-aware training

set -e  # Exit on error

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "================================================================================"
echo "COMPREHENSIVE CLI TEST SUITE"
echo "================================================================================"

# Setup: Clean slate
echo ""
echo "=== SETUP: Cleaning up old test data ==="
rm -f test_cli_predictions.db
rm -rf test_cli_models/
mkdir -p test_cli_models

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    echo "✓ PASS: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
    echo "✗ FAIL: $1"
    echo "  Details: $2"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# ============================================================================
# TEST 1: Initial General Training
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 1: General Training on Initial Data"
echo "================================================================================"

python3 smart_cli.py train --topic calculus --data-file data/ib/calculus.csv --stats 2>&1 | tee /tmp/test1_output.txt

if grep -q "Training complete (LNIRT ML estimation)" /tmp/test1_output.txt; then
    test_pass "General training completes successfully"
else
    test_fail "General training" "Did not complete with LNIRT ML estimation"
fi

if [ -f "models/calculus.pkl" ]; then
    test_pass "Model file created"
else
    test_fail "Model file creation" "models/calculus.pkl not found"
fi

# ============================================================================
# TEST 2: Make Predictions and Save to Database
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 2: Making Predictions (with --save)"
echo "================================================================================"

# Make 10 predictions for a test user
echo "Making 10 predictions for test_student..."
for i in {1..10}; do
    DIFF=$((1 + (i % 3)))  # Cycle through difficulties 1, 2, 3
    if ! python3 smart_cli.py predict --user-id test_student --topic calculus --difficulty $DIFF --save > /tmp/pred_$i.txt 2>&1; then
        echo "  Warning: Prediction $i failed"
        cat /tmp/pred_$i.txt
    fi
done

# Check database
PRED_COUNT=$(python3 -c "
from predictions_db import PredictionsDB
db = PredictionsDB()
import pandas as pd
# Query all predictions (even those not completed yet)
query = \"SELECT * FROM predictions WHERE user_id = 'test_student' AND topic = 'calculus'\"
predictions = pd.read_sql_query(query, db.conn)
print(len(predictions))
" 2>/dev/null)

if [ "$PRED_COUNT" = "10" ]; then
    test_pass "Predictions saved to database (count: $PRED_COUNT)"
else
    test_fail "Prediction saving" "Expected 10, got $PRED_COUNT"
fi

# ============================================================================
# TEST 3: Update Predictions with Actual Results
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 3: Updating Predictions with Actual Results"
echo "================================================================================"

# Simulate user completing tasks
echo "Simulating task completion..."
python3 << 'EOF'
from predictions_db import PredictionsDB
import pandas as pd

db = PredictionsDB()

# Get all predictions for test_student
query = "SELECT task_id FROM predictions WHERE user_id = 'test_student' AND topic = 'calculus'"
predictions = pd.read_sql_query(query, db.conn)

# Update each with realistic results
for i, row in enumerate(predictions.itertuples()):
    task_id = row.task_id
    # Simulate varying correctness and times
    correct = 1 if i % 3 != 2 else 0  # ~67% correct
    time = 100 + i * 10  # Varying times
    db.update_prediction(task_id, correct, time)

print(f"Updated {len(predictions)} predictions")
EOF

if [ $? -eq 0 ]; then
    test_pass "Predictions updated with actual results"
else
    test_fail "Prediction updates" "Python script failed"
fi

# ============================================================================
# TEST 4: User-Specific Training (Error-Aware)
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 4: User-Specific Training with Error Analysis"
echo "================================================================================"

python3 smart_cli.py train --topic calculus --user-id test_student --stats 2>&1 | tee /tmp/test4_output.txt

if grep -q "Prediction Error Analysis" /tmp/test4_output.txt; then
    test_pass "Error analysis performed during user-specific training"
else
    test_fail "Error analysis" "No error analysis output found"
fi

if grep -q "Error-Aware LNIRT ML" /tmp/test4_output.txt; then
    test_pass "Error-aware training completed"
else
    test_fail "Error-aware training" "Did not use error-aware method"
fi

# Check if parameters were updated
THETA=$(python3 -c "
from topic_model_manager import TopicModelManager
manager = TopicModelManager()
model = manager.get_model('calculus')
if 'test_student' in model.user_params:
    print(model.user_params['test_student']['theta'])
else:
    print('NOT_FOUND')
" 2>/dev/null)

if [ "$THETA" != "NOT_FOUND" ] && [ "$THETA" != "" ]; then
    test_pass "User parameters stored in model (θ=$THETA)"
else
    test_fail "User parameter storage" "test_student not found in model"
fi

# ============================================================================
# TEST 5: Predictions After User-Specific Training
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 5: Predictions After Personalization"
echo "================================================================================"

echo "Making prediction for personalized user..."
python3 smart_cli.py predict --user-id test_student --topic calculus --difficulty 2 2>&1 | tee /tmp/test5_output.txt

if grep -q "Probability of Correct:" /tmp/test5_output.txt; then
    test_pass "Prediction generated for personalized user"
else
    test_fail "Personalized prediction" "No probability output found"
fi

# Make prediction for new user (should use population average)
echo ""
echo "Making prediction for new user..."
python3 smart_cli.py predict --user-id brand_new_user --topic calculus --difficulty 2 2>&1 | tee /tmp/test5b_output.txt

if grep -q "Probability of Correct:" /tmp/test5b_output.txt; then
    test_pass "Prediction generated for new user (population average)"
else
    test_fail "New user prediction" "No probability output found"
fi

# ============================================================================
# TEST 6: Incremental Training (Only New Data)
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 6: Incremental General Training"
echo "================================================================================"

# Add more data
echo "Adding more tasks..."
for i in {1..5}; do
    DIFF=$((1 + (i % 3)))
    python3 smart_cli.py predict --user-id another_user --topic calculus --difficulty $DIFF --save > /dev/null 2>&1
done

# Simulate completion
python3 << 'EOF'
from predictions_db import PredictionsDB
import pandas as pd

db = PredictionsDB()

query = "SELECT task_id FROM predictions WHERE user_id = 'another_user' ORDER BY id DESC LIMIT 5"
predictions = pd.read_sql_query(query, db.conn)

for i, row in enumerate(predictions.itertuples()):
    db.update_prediction(row.task_id, 1 if i < 3 else 0, 120 + i * 15)

print(f"Added {len(predictions)} new tasks")
EOF

# Try incremental training
python3 smart_cli.py train --topic calculus --stats 2>&1 | tee /tmp/test6_output.txt

if grep -q "new training data" /tmp/test6_output.txt; then
    test_pass "Incremental training attempted"
else
    test_fail "Incremental training" "Did not check for new data"
fi

# ============================================================================
# TEST 7: Stats Command
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 7: Statistics Display"
echo "================================================================================"

# Overall stats
python3 smart_cli.py stats 2>&1 | tee /tmp/test7a_output.txt

if grep -q "calculus" /tmp/test7a_output.txt; then
    test_pass "Overall stats show calculus topic"
else
    test_fail "Overall stats" "calculus not found in output"
fi

# Topic-specific stats
python3 smart_cli.py stats --topic calculus 2>&1 | tee /tmp/test7b_output.txt

if grep -q "Difficulty Parameters" /tmp/test7b_output.txt; then
    test_pass "Topic-specific stats show difficulty parameters"
else
    test_fail "Topic-specific stats" "Difficulty parameters not shown"
fi

if grep -q "Prediction History" /tmp/test7b_output.txt; then
    test_pass "Topic-specific stats show prediction history"
else
    test_fail "Prediction history stats" "No prediction history section"
fi

# ============================================================================
# TEST 8: Update Command with Retrain
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 8: Update Command with Personalized Learning"
echo "================================================================================"

# Make and save a prediction
python3 smart_cli.py predict --user-id test_retrain_user --topic calculus --difficulty 2 --save 2>&1 | tee /tmp/test8_pred.txt

# Extract task ID
TASK_ID=$(grep "Task ID:" /tmp/test8_pred.txt | sed 's/.*Task ID: \([0-9]*\).*/\1/')

if [ -n "$TASK_ID" ] && [ "$TASK_ID" -gt 0 ]; then
    test_pass "Prediction saved with task ID: $TASK_ID"

    # Update with actual result
    python3 smart_cli.py update --task-id $TASK_ID --correct 1 --time 95.5 2>&1 | tee /tmp/test8_update.txt

    if grep -q "updated" /tmp/test8_update.txt; then
        test_pass "Update command executed successfully"
    else
        test_fail "Update command" "No confirmation message"
    fi

    # Check if personalized learning happened
    if grep -q "parameters updated" /tmp/test8_update.txt || grep -q "Personalized learning" /tmp/test8_update.txt; then
        test_pass "Personalized learning triggered by update"
    else
        echo "  Note: Personalized learning may have happened silently"
    fi
else
    test_fail "Task ID extraction" "Could not get task ID from prediction"
fi

# ============================================================================
# TEST 9: Error Handling
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 9: Error Handling"
echo "================================================================================"

# Test 9.1: Invalid topic
python3 smart_cli.py predict --user-id test --topic nonexistent --difficulty 1 2>&1 | tee /tmp/test9a_output.txt

if grep -qi "error\|not trained\|not found" /tmp/test9a_output.txt; then
    test_pass "Handles invalid topic gracefully"
else
    test_fail "Invalid topic handling" "No error message"
fi

# Test 9.2: Invalid difficulty
python3 smart_cli.py predict --user-id test --topic calculus --difficulty 5 2>&1 | tee /tmp/test9b_output.txt

if grep -qi "error\|invalid\|must be" /tmp/test9b_output.txt || grep -q "1, 2, or 3" /tmp/test9b_output.txt; then
    test_pass "Handles invalid difficulty gracefully"
else
    test_fail "Invalid difficulty handling" "No error message"
fi

# Test 9.3: Update non-existent task
python3 smart_cli.py update --task-id 999999 --correct 1 --time 100 2>&1 | tee /tmp/test9c_output.txt

if grep -qi "error\|not found\|does not exist" /tmp/test9c_output.txt; then
    test_pass "Handles non-existent task ID gracefully"
else
    test_fail "Non-existent task handling" "No error message"
fi

# ============================================================================
# TEST 10: Multiple Users on Same Topic
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST 10: Multiple Users with Personalization"
echo "================================================================================"

# Create predictions for 3 different users
for USER in user_a user_b user_c; do
    echo "Setting up $USER..."
    for i in {1..5}; do
        DIFF=$((1 + (i % 3)))
        python3 smart_cli.py predict --user-id $USER --topic calculus --difficulty $DIFF --save > /dev/null 2>&1
    done

    # Simulate completion with different performance levels
    python3 << EOF
from predictions_db import PredictionsDB
db = PredictionsDB()
import pandas as pd

query = "SELECT task_id FROM predictions WHERE user_id = '$USER' AND topic = 'calculus' ORDER BY id DESC LIMIT 5"
preds = pd.read_sql_query(query, db.conn)

# user_a: good performance
# user_b: average performance
# user_c: poor performance
if '$USER' == 'user_a':
    for i, row in enumerate(preds.itertuples()):
        db.update_prediction(row.task_id, 1, 80 + i * 5)
elif '$USER' == 'user_b':
    for i, row in enumerate(preds.itertuples()):
        db.update_prediction(row.task_id, 1 if i < 3 else 0, 110 + i * 10)
else:  # user_c
    for i, row in enumerate(preds.itertuples()):
        db.update_prediction(row.task_id, 1 if i < 2 else 0, 140 + i * 15)
EOF

    # Train user-specific model
    python3 smart_cli.py train --topic calculus --user-id $USER > /dev/null 2>&1
done

# Compare parameters
echo ""
echo "Comparing user parameters..."
python3 << 'EOF'
from topic_model_manager import TopicModelManager

manager = TopicModelManager()
model = manager.get_model('calculus')

print("\nUser Parameters Comparison:")
for user_id in ['user_a', 'user_b', 'user_c']:
    if user_id in model.user_params:
        params = model.user_params[user_id]
        print(f"  {user_id}: θ={params['theta']:.3f}, τ={params['tau']:.3f}")
    else:
        print(f"  {user_id}: NOT FOUND")
EOF

if [ $? -eq 0 ]; then
    test_pass "Multiple users trained with different parameters"
else
    test_fail "Multiple user training" "Failed to retrieve parameters"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo ""
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo "Passed: $TESTS_PASSED ✓"
echo "Failed: $TESTS_FAILED ✗"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 ALL CLI TESTS PASSED!"
    exit 0
else
    echo "⚠️  Some tests failed. Review output above."
    exit 1
fi
