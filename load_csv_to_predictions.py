#!/usr/bin/env python3
"""
Load CSV training data into predictions table with BOTH predicted and actual values.

This simulates the real SmartStudy workflow:
1. Initial model makes predictions (predicted_correct, predicted_time)
2. User answers question (actual_correct, actual_time)
3. Both are stored in predictions table

For CSV data, we need to retroactively generate what the predictions would have been.
"""

import pandas as pd
import numpy as np
from predictions_db import PredictionsDB
from topic_lnirt import TopicLNIRTModel
from datetime import datetime, timedelta
import sys

def load_csv_with_predictions(csv_path: str, topic: str, simulate_timestamps: bool = True, force: bool = False):
    """
    Load CSV data and populate predictions table with retroactive predictions.

    Strategy:
    1. Load CSV data (has actual results)
    2. Train an initial "naive" model on a subset to get baseline
    3. For each record, generate what the prediction would have been
    4. Insert into predictions table with both predicted and actual

    Args:
        csv_path: Path to CSV file
        topic: Topic name
        simulate_timestamps: If True, spread timestamps over past days
        force: If True, delete existing data without asking
    """
    print("=" * 80)
    print("LOADING CSV DATA TO PREDICTIONS TABLE")
    print("=" * 80)

    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    data = pd.read_csv(csv_path)
    print(f"  Loaded {len(data)} records")
    print(f"  Users: {data['user_id'].nunique()}")
    print(f"  Difficulties: {sorted(data['difficulty'].unique())}")

    # Validate columns
    required_cols = ['user_id', 'difficulty', 'correct', 'response_time']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    # Initialize database
    db = PredictionsDB()

    # Check if topic already has data
    query = f"SELECT COUNT(*) as count FROM predictions WHERE topic = '{topic}'"
    existing_count = pd.read_sql_query(query, db.conn)['count'][0]

    if existing_count > 0:
        print(f"\n⚠ Warning: Topic '{topic}' already has {existing_count} predictions")
        if not force:
            response = input("Delete existing data and reload? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborting.")
                sys.exit(0)
        else:
            print("  Force mode: deleting existing data...")

        # Delete existing data
        db.conn.execute(f"DELETE FROM predictions WHERE topic = '{topic}'")
        db.conn.execute(f"DELETE FROM training_data WHERE topic = '{topic}'")
        db.conn.commit()
        print(f"  Deleted {existing_count} existing predictions")

    # Strategy: Use a simple baseline model to generate retroactive predictions
    # We'll use population statistics from the data itself

    print("\n" + "=" * 80)
    print("GENERATING RETROACTIVE PREDICTIONS")
    print("=" * 80)

    # Calculate baseline statistics by difficulty
    baseline_stats = {}
    for diff in [1, 2, 3]:
        diff_data = data[data['difficulty'] == diff]
        if len(diff_data) > 0:
            baseline_stats[diff] = {
                'accuracy': diff_data['correct'].mean(),
                'mean_time': diff_data['response_time'].mean(),
                'std_time': diff_data['response_time'].std()
            }
            print(f"\nDifficulty {diff} baseline:")
            print(f"  Accuracy: {baseline_stats[diff]['accuracy']:.1%}")
            print(f"  Mean time: {baseline_stats[diff]['mean_time']:.1f}s")
            print(f"  Std time: {baseline_stats[diff]['std_time']:.1f}s")

    # For more realistic predictions, we'll add some noise
    # and adjust based on user's historical performance

    print("\n" + "=" * 80)
    print("POPULATING PREDICTIONS TABLE")
    print("=" * 80)

    # Calculate user-level statistics
    user_stats = {}
    for user_id in data['user_id'].unique():
        user_data = data[data['user_id'] == user_id]
        user_stats[user_id] = {
            'overall_accuracy': user_data['correct'].mean(),
            'avg_time': user_data['response_time'].mean()
        }

    # Generate predictions and insert
    base_timestamp = datetime.now() - timedelta(days=30)  # Start 30 days ago
    records_inserted = 0

    for idx, row in data.iterrows():
        user_id = row['user_id']
        difficulty = int(row['difficulty'])
        actual_correct = int(row['correct'])
        actual_time = float(row['response_time'])

        # Generate retroactive prediction
        # Base on difficulty baseline, adjusted by user's overall performance
        base_accuracy = baseline_stats[difficulty]['accuracy']
        user_adjustment = (user_stats[user_id]['overall_accuracy'] - 0.5) * 0.3
        predicted_correct = np.clip(base_accuracy + user_adjustment + np.random.normal(0, 0.1), 0.1, 0.9)

        base_time = baseline_stats[difficulty]['mean_time']
        user_time_ratio = user_stats[user_id]['avg_time'] / base_time if base_time > 0 else 1.0
        time_noise = np.random.normal(1.0, 0.15)
        predicted_time = base_time * user_time_ratio * time_noise
        predicted_time = max(10.0, predicted_time)  # At least 10 seconds

        # Generate timestamp (spread over past 30 days)
        if simulate_timestamps:
            timestamp = base_timestamp + timedelta(seconds=idx * 1800)  # 30 min apart
        else:
            timestamp = datetime.now()

        # Insert into predictions table
        cursor = db.conn.cursor()
        cursor.execute('''
        INSERT INTO predictions (user_id, topic, difficulty, predicted_correct, predicted_time,
                                actual_correct, actual_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, topic, difficulty, predicted_correct, predicted_time,
              actual_correct, actual_time, timestamp.isoformat(), timestamp.isoformat()))

        records_inserted += 1

        if (idx + 1) % 300 == 0:
            print(f"  Inserted {idx + 1}/{len(data)} records...")

    db.conn.commit()

    print(f"\n✓ Inserted {records_inserted} records into predictions table")

    # Also populate training_data table for general training
    print("\n" + "=" * 80)
    print("POPULATING TRAINING_DATA TABLE")
    print("=" * 80)

    for idx, row in data.iterrows():
        cursor = db.conn.cursor()
        timestamp = base_timestamp + timedelta(seconds=idx * 1800) if simulate_timestamps else datetime.now()
        cursor.execute('''
        INSERT INTO training_data (user_id, topic, difficulty, correct, response_time, timestamp, used_for_general_training)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row['user_id'], topic, int(row['difficulty']), int(row['correct']),
              float(row['response_time']), timestamp.isoformat(), 0))

    db.conn.commit()
    print(f"✓ Inserted {len(data)} records into training_data table")

    # Verify
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    pred_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM predictions WHERE topic='{topic}'", db.conn)['count'][0]
    train_count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM training_data WHERE topic='{topic}'", db.conn)['count'][0]

    print(f"\nPredictions table: {pred_count} records")
    print(f"Training_data table: {train_count} records")

    # Check that all have both predicted and actual
    complete_query = f'''
    SELECT COUNT(*) as count
    FROM predictions
    WHERE topic='{topic}' AND predicted_correct IS NOT NULL AND actual_correct IS NOT NULL
    '''
    complete_count = pd.read_sql_query(complete_query, db.conn)['count'][0]
    print(f"Complete records (predicted+actual): {complete_count}")

    if complete_count == pred_count:
        print("✓ All records have both predicted and actual data!")
    else:
        print(f"⚠ Warning: {pred_count - complete_count} records missing data")

    # Sample check
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)

    sample_query = f'''
    SELECT user_id, difficulty, predicted_correct, predicted_time, actual_correct, actual_time
    FROM predictions
    WHERE topic='{topic}'
    LIMIT 5
    '''
    sample = pd.read_sql_query(sample_query, db.conn)
    print("\nFirst 5 records:")
    print(sample.to_string(index=False))

    db.close()

    print("\n" + "=" * 80)
    print("✓ CSV DATA LOADED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll {records_inserted} records now have:")
    print("  ✓ Predicted values (retroactively generated)")
    print("  ✓ Actual values (from CSV)")
    print("  ✓ Ready for error-aware training!")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 load_csv_to_predictions.py <csv_path> <topic> [--force]")
        print("Example: python3 load_csv_to_predictions.py data/ib/calculus.csv calculus --force")
        sys.exit(1)

    csv_path = sys.argv[1]
    topic = sys.argv[2]
    force = '--force' in sys.argv

    load_csv_with_predictions(csv_path, topic, force=force)
