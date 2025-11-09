#!/usr/bin/env python3
"""
SmartStudy LNIRT CLI
Simplified interface for topic-based predictions
"""

import argparse
import sys
from topic_lnirt import TopicModelManager
from predictions_db import PredictionsDB


def train_command(args):
    """Train model on topic-specific data"""
    import pandas as pd

    print("=" * 70)
    if args.user_id:
        print(f"TRAINING MODEL (USER-SPECIFIC): {args.topic} for {args.user_id}")
    else:
        print(f"TRAINING MODEL (GENERAL): {args.topic}")
    print("=" * 70)
    print()

    manager = TopicModelManager()
    model = manager.get_model(args.topic)
    db = PredictionsDB()

    if args.user_id:
        # USER-SPECIFIC TRAINING
        print(f"Loading user-specific training data for {args.user_id}...")

        # Get user's data from predictions table (includes both predictions and actuals)
        user_data = db.get_user_training_data(args.user_id, args.topic)

        if len(user_data) == 0:
            print(f"Warning: No completed tasks found for user {args.user_id} in topic {args.topic}")
            print("User needs to complete at least one task before personalized training")
            db.close()
            sys.exit(1)

        print(f"  Found {len(user_data)} completed tasks for this user")
        print(f"  Accuracy: {user_data['correct'].mean():.1%}")
        print(f"  Mean time: {user_data['response_time'].mean():.1f}s")
        print()

        # Train user-specific parameters
        print("Training personalized model for this user...")
        model.fit_user_specific(user_data, args.user_id, verbose=True)
        manager.save_model(args.topic)

        # Also contribute actual results to general training pool
        print()
        print("Contributing user's actual results to general training pool...")
        # This data is already in training_data table from update_prediction

        print()
        print("=" * 70)
        print("USER-SPECIFIC TRAINING COMPLETE!")
        print("=" * 70)

    else:
        # GENERAL TRAINING (all users, new data only)
        print("Loading new training data from database...")

        # Get only new data not yet used for general training
        new_data = db.get_training_data(args.topic, only_new=True)

        if len(new_data) == 0:
            print("No new training data found in database.")

            # Try loading from file if provided
            if args.data_file:
                print(f"Loading from file: {args.data_file}")
                try:
                    data = pd.read_csv(args.data_file)

                    # Validate columns
                    required_cols = ['user_id', 'difficulty', 'correct', 'response_time']
                    missing = [c for c in required_cols if c not in data.columns]
                    if missing:
                        print(f"Error: Missing columns: {missing}")
                        print(f"Required: {required_cols}")
                        db.close()
                        sys.exit(1)

                    # Validate difficulty values
                    if not data['difficulty'].isin([1, 2, 3]).all():
                        print("Error: Difficulty must be 1, 2, or 3")
                        db.close()
                        sys.exit(1)

                    print(f"Loaded {len(data)} training samples from file")
                    print(f"  Users: {data['user_id'].nunique()}")
                    print(f"  Difficulties: {sorted(data['difficulty'].unique())}")
                    print(f"  Accuracy: {data['correct'].mean():.1%}")
                    print(f"  Mean time: {data['response_time'].mean():.1f}s")
                    print()

                except FileNotFoundError:
                    print(f"Error: File not found: {args.data_file}")
                    db.close()
                    sys.exit(1)
            else:
                print("No new data and no file provided. Nothing to train.")
                db.close()
                sys.exit(1)
        else:
            print(f"  Found {len(new_data)} new training samples")
            print(f"  Users: {new_data['user_id'].nunique()}")
            print(f"  Accuracy: {new_data['correct'].mean():.1%}")
            print(f"  Mean time: {new_data['response_time'].mean():.1f}s")
            print()
            data = new_data

        # Train model
        print("Training general model on new data...")
        model.fit(data, verbose=True)
        manager.save_model(args.topic)

        # Mark data as used
        db.mark_training_data_used(args.topic)

        print()
        print("=" * 70)
        print("GENERAL TRAINING COMPLETE!")
        print("=" * 70)

    db.close()

    # Show model stats
    if args.stats:
        stats = model.get_stats()
        print()
        print("MODEL STATISTICS:")
        print(f"  Topic: {stats['topic']}")
        print(f"  Users trained: {stats['n_users']}")
        print()
        print("Difficulty Parameters:")
        for diff in [1, 2, 3]:
            params = stats['difficulty_params'][diff]
            print(f"  Level {diff}: difficulty={params['b']:.2f}, time_intensity={params['beta']:.2f}")

        if 'user_ability' in stats:
            print()
            print(f"User Ability: mean={stats['user_ability']['mean']:.2f}, "
                  f"range=[{stats['user_ability']['min']:.2f}, {stats['user_ability']['max']:.2f}]")
            print(f"User Speed: mean={stats['user_speed']['mean']:.2f}, "
                  f"range=[{stats['user_speed']['min']:.2f}, {stats['user_speed']['max']:.2f}]")


def predict_command(args):
    """Make prediction for user on topic/difficulty"""
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)
    print()

    # Load model
    manager = TopicModelManager()
    model = manager.get_model(args.topic)

    if not model.is_trained:
        print(f"Error: Model for topic '{args.topic}' is not trained")
        print(f"Train it first with: python3 smart_cli.py train --topic {args.topic} --data-file data/ib/{args.topic}.csv")
        sys.exit(1)

    # Make prediction
    try:
        p_correct, expected_time = model.predict(args.user_id, args.difficulty)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"User: {args.user_id}")
    print(f"Topic: {args.topic}")
    print(f"Difficulty: {args.difficulty}")
    print()
    print("-" * 70)
    print("PREDICTION RESULTS")
    print("-" * 70)
    print(f"Probability of Correct: {p_correct:.1%}")
    print(f"Expected Time: {expected_time:.1f} seconds ({expected_time/60:.1f} minutes)")
    print()

    # Interpretation
    if p_correct >= 0.7:
        confidence = "HIGH"
        emoji = "✓"
    elif p_correct >= 0.5:
        confidence = "MEDIUM"
        emoji = "~"
    else:
        confidence = "LOW"
        emoji = "✗"

    print(f"{emoji} Confidence: {confidence} ({p_correct:.0%})")

    if expected_time < 60:
        print(f"⚡ Expected to complete quickly ({expected_time:.0f}s)")
    elif expected_time < 180:
        print(f"⏱ Moderate time expected ({expected_time/60:.1f} min)")
    else:
        print(f"⏳ Significant time required ({expected_time/60:.1f} min)")

    # Save prediction to database
    if args.save:
        db = PredictionsDB()
        task_id = db.add_prediction(args.user_id, args.topic, args.difficulty, p_correct, expected_time)
        db.close()
        print()
        print(f"✓ Prediction saved with task_id={task_id}")
        print(f"  Update later with: python3 smart_cli.py update --task-id {task_id} --correct [0/1] --time [seconds]")

    print()
    print("=" * 70)


def update_command(args):
    """Update prediction with actual results and retrain model"""
    print("=" * 70)
    print("UPDATE PREDICTION WITH ACTUAL RESULTS")
    print("=" * 70)
    print()

    # Get prediction
    db = PredictionsDB()
    prediction = db.get_prediction(args.task_id)

    if not prediction:
        print(f"Error: Task ID {args.task_id} not found")
        db.close()
        sys.exit(1)

    print(f"Task ID: {args.task_id}")
    print(f"User: {prediction['user_id']}")
    print(f"Topic: {prediction['topic']}")
    print(f"Difficulty: {prediction['difficulty']}")
    print()
    print("Predicted:")
    print(f"  Correctness: {prediction['predicted_correct']:.1%}")
    print(f"  Time: {prediction['predicted_time']:.1f}s")
    print()
    print("Actual:")
    print(f"  Correctness: {'CORRECT' if args.correct == 1 else 'INCORRECT'}")
    print(f"  Time: {args.time:.1f}s")
    print()

    # Update database
    db.update_prediction(args.task_id, args.correct, args.time)
    print("✓ Database updated")

    # Update model (dual learning)
    print()
    print("Updating model...")

    manager = TopicModelManager()
    model = manager.get_model(prediction['topic'])

    # Personalized learning: update user-specific parameters
    model.update_from_response(
        prediction['user_id'],
        prediction['difficulty'],
        args.correct,
        args.time
    )

    manager.save_model(prediction['topic'])
    print("✓ Model updated (personalized learning applied)")

    # General learning: retrain on accumulated data if enough new data
    training_data = db.get_training_data(prediction['topic'])
    if len(training_data) >= args.retrain_threshold:
        print(f"✓ Retraining model on {len(training_data)} accumulated responses...")
        model.fit(training_data, verbose=False)
        manager.save_model(prediction['topic'])
        print("✓ General model parameters updated")
    else:
        print(f"  (General retrain will occur after {args.retrain_threshold - len(training_data)} more responses)")

    db.close()

    print()
    print("=" * 70)
    print("UPDATE COMPLETE!")
    print("=" * 70)


def stats_command(args):
    """Display statistics"""
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print()

    if args.topic:
        # Topic-specific stats
        manager = TopicModelManager()
        model = manager.get_model(args.topic)

        if not model.is_trained:
            print(f"Model for topic '{args.topic}' is not trained yet")
            return

        stats = model.get_stats()

        print(f"Topic: {stats['topic']}")
        print(f"Users: {stats['n_users']}")
        print()

        print("Difficulty Parameters:")
        for diff in [1, 2, 3]:
            params = stats['difficulty_params'][diff]
            print(f"  Level {diff}:")
            print(f"    Discrimination (a): {params['a']:.2f}")
            print(f"    Difficulty (b): {params['b']:.2f}")
            print(f"    Time Intensity (β): {params['beta']:.2f} (≈{np.exp(params['beta']):.0f}s)")

        if 'user_ability' in stats:
            print()
            print("User Statistics:")
            print(f"  Ability (θ):")
            print(f"    Mean: {stats['user_ability']['mean']:.2f}")
            print(f"    Std: {stats['user_ability']['std']:.2f}")
            print(f"    Range: [{stats['user_ability']['min']:.2f}, {stats['user_ability']['max']:.2f}]")
            print(f"  Speed (τ):")
            print(f"    Mean: {stats['user_speed']['mean']:.2f}")
            print(f"    Std: {stats['user_speed']['std']:.2f}")
            print(f"    Range: [{stats['user_speed']['min']:.2f}, {stats['user_speed']['max']:.2f}]")

        # Database stats
        db = PredictionsDB()
        db_stats = db.get_topic_stats(args.topic)
        db.close()

        if db_stats['total_predictions'] > 0:
            print()
            print("Prediction History:")
            print(f"  Total predictions: {db_stats['total_predictions']}")
            print(f"  Completed: {db_stats['completed']}")

            print()
            print("Actual Performance by Difficulty:")
            for diff in [1, 2, 3]:
                acc = db_stats['accuracy_by_difficulty'][diff]
                time = db_stats['avg_time_by_difficulty'][diff]
                if acc is not None:
                    print(f"  Level {diff}: accuracy={acc:.1%}, avg_time={time:.1f}s")

    else:
        # List all topics
        manager = TopicModelManager()
        topics = manager.list_topics()

        if not topics:
            print("No trained models found")
            print("Train a model with: python3 smart_cli.py train --topic <topic> --data-file <file>")
            return

        print(f"Available Topics ({len(topics)}):")
        for topic in topics:
            model = manager.get_model(topic)
            stats = model.get_stats()
            print(f"  - {topic:20s} ({stats['n_users']} users trained)")

        print()
        print("Use --topic <name> to see detailed statistics")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="SmartStudy LNIRT CLI - Topic-based predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on topic data')
    train_parser.add_argument('--topic', required=True, help='Topic name')
    train_parser.add_argument('--data-file', help='Path to training CSV (optional if training from database)')
    train_parser.add_argument('--user-id', help='Train for specific user only (uses prediction history)')
    train_parser.add_argument('--stats', action='store_true', help='Show statistics after training')
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict for user on topic/difficulty')
    predict_parser.add_argument('--user-id', required=True, help='User ID')
    predict_parser.add_argument('--topic', required=True, help='Topic name')
    predict_parser.add_argument('--difficulty', type=int, required=True, choices=[1, 2, 3],
                                help='Difficulty level (1=easy, 2=medium, 3=hard)')
    predict_parser.add_argument('--save', action='store_true', help='Save prediction to database')
    predict_parser.set_defaults(func=predict_command)

    # Update command
    update_parser = subparsers.add_parser('update', help='Update prediction with actual results')
    update_parser.add_argument('--task-id', type=int, required=True, help='Task ID from prediction')
    update_parser.add_argument('--correct', type=int, required=True, choices=[0, 1],
                               help='1 if correct, 0 if incorrect')
    update_parser.add_argument('--time', type=float, required=True, help='Actual time in seconds')
    update_parser.add_argument('--retrain-threshold', type=int, default=50,
                               help='Retrain model after N new responses (default: 50)')
    update_parser.set_defaults(func=update_command)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display statistics')
    stats_parser.add_argument('--topic', help='Show stats for specific topic')
    stats_parser.set_defaults(func=stats_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Import numpy here to avoid loading it for help messages
    import numpy as np
    globals()['np'] = np

    args.func(args)


if __name__ == '__main__':
    main()
