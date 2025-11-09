#!/usr/bin/env python3
"""
LNIRT Model CLI

Command-line interface for training and using the LNIRT model for:
- Predicting task completion time
- Predicting probability of correct response
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path
from lnirt_model import LNIRTModel


def train_command(args):
    """Train the LNIRT model on provided data"""
    print("=" * 70)
    print("TRAINING LNIRT MODEL")
    print("=" * 70)

    # Load training data
    print(f"\nLoading training data from: {args.data_file}")
    try:
        data = pd.read_csv(args.data_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.data_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Validate required columns
    required_cols = ['user_id', 'item_id', 'correct', 'response_time']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        print("\nRequired format:")
        print("  - user_id: unique identifier for each user/student")
        print("  - item_id: unique identifier for each question/task")
        print("  - correct: 1 for correct answer, 0 for incorrect")
        print("  - response_time: time in seconds (can be decimal)")
        sys.exit(1)

    # Remove any rows with missing values
    original_len = len(data)
    data = data[required_cols].dropna()
    if len(data) < original_len:
        print(f"Warning: Removed {original_len - len(data)} rows with missing values")

    # Validate data
    if len(data) == 0:
        print("Error: No valid data rows found")
        sys.exit(1)

    if not data['correct'].isin([0, 1]).all():
        print("Error: 'correct' column must contain only 0 or 1")
        sys.exit(1)

    if (data['response_time'] <= 0).any():
        print("Error: 'response_time' must be positive")
        sys.exit(1)

    print(f"\nTraining data loaded successfully:")
    print(f"  - Total responses: {len(data)}")
    print(f"  - Unique users: {data['user_id'].nunique()}")
    print(f"  - Unique items: {data['item_id'].nunique()}")
    print(f"  - Overall accuracy: {data['correct'].mean():.1%}")
    print(f"  - Mean response time: {data['response_time'].mean():.1f} seconds")

    # Initialize and train model
    model = LNIRTModel()

    print("\nTraining model (this may take a few minutes)...")
    model.fit(data, max_iter=args.max_iter, verbose=args.verbose)

    # Save model
    model_path = args.model_file
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    # Display summary statistics
    if args.stats:
        print("\n" + "=" * 70)
        print("MODEL STATISTICS")
        print("=" * 70)

        print("\nUser Parameters (sample):")
        user_stats = model.get_user_stats()
        print(user_stats.head(10).to_string(index=False))
        print(f"\nAbility (theta) - Mean: {user_stats['ability_theta'].mean():.2f}, "
              f"SD: {user_stats['ability_theta'].std():.2f}")
        print(f"Speed (tau) - Mean: {user_stats['speed_tau'].mean():.2f}, "
              f"SD: {user_stats['speed_tau'].std():.2f}")

        print("\nItem Parameters (sample):")
        item_stats = model.get_item_stats()
        print(item_stats.head(10).to_string(index=False))
        print(f"\nDiscrimination (a) - Mean: {item_stats['discrimination_a'].mean():.2f}, "
              f"SD: {item_stats['discrimination_a'].std():.2f}")
        print(f"Difficulty (b) - Mean: {item_stats['difficulty_b'].mean():.2f}, "
              f"SD: {item_stats['difficulty_b'].std():.2f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


def predict_command(args):
    """Make predictions for a new task"""
    print("=" * 70)
    print("LNIRT MODEL PREDICTION")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {args.model_file}")
    model = LNIRTModel()
    try:
        model.load_model(args.model_file)
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model_file}")
        print("Please train a model first using: python3 cli.py train --data-file <data.csv>")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Model loaded successfully!")

    # Get user and item IDs
    user_id = args.user_id
    item_id = args.item_id

    # Handle item features if provided
    item_features = None
    if args.item_features:
        try:
            item_features = json.loads(args.item_features)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for item features: {args.item_features}")
            sys.exit(1)

    # Make prediction
    print(f"\nMaking prediction for:")
    print(f"  User ID: {user_id}")
    print(f"  Item ID: {item_id}")
    if item_features:
        print(f"  Item features: {item_features}")

    try:
        p_correct, expected_time = model.predict(user_id, item_id, item_features)
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)

    # Display results
    print("\n" + "-" * 70)
    print("PREDICTION RESULTS")
    print("-" * 70)
    print(f"\nProbability of Correct Answer: {p_correct:.1%}")
    print(f"Expected Response Time: {expected_time:.1f} seconds ({expected_time/60:.1f} minutes)")

    # Provide interpretation
    print("\nInterpretation:")
    if p_correct >= 0.7:
        print(f"  ✓ High confidence ({p_correct:.0%}) - User likely to answer correctly")
    elif p_correct >= 0.5:
        print(f"  ~ Moderate confidence ({p_correct:.0%}) - User has a decent chance")
    else:
        print(f"  ✗ Low confidence ({p_correct:.0%}) - User likely to struggle")

    if expected_time < 60:
        print(f"  ⚡ Fast ({expected_time:.0f}s) - User expected to answer quickly")
    elif expected_time < 180:
        print(f"  ⏱ Moderate ({expected_time/60:.1f}m) - Average time expected")
    else:
        print(f"  ⏳ Slow ({expected_time/60:.1f}m) - User may need more time")

    # Check if user/item is new
    if user_id not in model.user_params:
        print(f"\n⚠ Note: '{user_id}' is a new user (not in training data)")
        print("  Prediction uses population average parameters")

    if item_id not in model.item_params:
        print(f"\n⚠ Note: '{item_id}' is a new item (not in training data)")
        if item_features:
            print("  Prediction uses provided item features")
        else:
            print("  Prediction uses population average parameters")

    print("\n" + "=" * 70)


def stats_command(args):
    """Display model statistics"""
    print("=" * 70)
    print("LNIRT MODEL STATISTICS")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {args.model_file}")
    model = LNIRTModel()
    try:
        model.load_model(args.model_file)
    except FileNotFoundError:
        print(f"Error: Model file not found: {args.model_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Display user statistics
    print("\n" + "-" * 70)
    print("USER PARAMETERS")
    print("-" * 70)
    user_stats = model.get_user_stats()
    print(f"\nTotal users: {len(user_stats)}")
    print(f"\nAbility (theta):")
    print(f"  Mean: {user_stats['ability_theta'].mean():.3f}")
    print(f"  Std Dev: {user_stats['ability_theta'].std():.3f}")
    print(f"  Range: [{user_stats['ability_theta'].min():.3f}, {user_stats['ability_theta'].max():.3f}]")
    print(f"\nSpeed (tau):")
    print(f"  Mean: {user_stats['speed_tau'].mean():.3f}")
    print(f"  Std Dev: {user_stats['speed_tau'].std():.3f}")
    print(f"  Range: [{user_stats['speed_tau'].min():.3f}, {user_stats['speed_tau'].max():.3f}]")

    if args.top_users:
        print(f"\nTop {args.top_users} users by ability:")
        top_users = user_stats.nlargest(args.top_users, 'ability_theta')
        print(top_users.to_string(index=False))

    # Display item statistics
    print("\n" + "-" * 70)
    print("ITEM PARAMETERS")
    print("-" * 70)
    item_stats = model.get_item_stats()
    print(f"\nTotal items: {len(item_stats)}")
    print(f"\nDiscrimination (a):")
    print(f"  Mean: {item_stats['discrimination_a'].mean():.3f}")
    print(f"  Std Dev: {item_stats['discrimination_a'].std():.3f}")
    print(f"\nDifficulty (b):")
    print(f"  Mean: {item_stats['difficulty_b'].mean():.3f}")
    print(f"  Std Dev: {item_stats['difficulty_b'].std():.3f}")
    print(f"\nTime Intensity (beta):")
    print(f"  Mean: {item_stats['time_intensity_beta'].mean():.3f}")
    print(f"  Std Dev: {item_stats['time_intensity_beta'].std():.3f}")

    if args.hardest_items:
        print(f"\nHardest {args.hardest_items} items (by difficulty):")
        hardest = item_stats.nlargest(args.hardest_items, 'difficulty_b')
        print(hardest.to_string(index=False))

    # Global parameters
    print("\n" + "-" * 70)
    print("GLOBAL PARAMETERS")
    print("-" * 70)
    print(f"Response time residual SD (sigma): {model.sigma:.3f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="LNIRT Model CLI - Train and predict with joint IRT + response time model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on sample data
  python3 cli.py train --data-file data/sample_training_data.csv

  # Train with custom options
  python3 cli.py train --data-file mydata.csv --model-file my_model.pkl --stats

  # Make prediction for a user on a task
  python3 cli.py predict --user-id user_001 --item-id item_050

  # Predict for new item with estimated features
  python3 cli.py predict --user-id user_001 --item-id new_item \\
    --item-features '{"a": 1.5, "b": 0.5, "beta": 4.0}'

  # View model statistics
  python3 cli.py stats --top-users 10 --hardest-items 10

For more information, see README.md
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model on data')
    train_parser.add_argument('--data-file', '-d', required=True,
                              help='Path to CSV file with training data')
    train_parser.add_argument('--model-file', '-m', default='models/lnirt_model.pkl',
                              help='Path to save trained model (default: models/lnirt_model.pkl)')
    train_parser.add_argument('--max-iter', type=int, default=1000,
                              help='Maximum optimization iterations (default: 1000)')
    train_parser.add_argument('--stats', action='store_true',
                              help='Display model statistics after training')
    train_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Show detailed training progress')
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions for a task')
    predict_parser.add_argument('--user-id', '-u', required=True,
                                help='User ID to make prediction for')
    predict_parser.add_argument('--item-id', '-i', required=True,
                                help='Item/task ID to make prediction for')
    predict_parser.add_argument('--item-features', '-f',
                                help='JSON string with item features (for new items): '
                                     '{"a": discrimination, "b": difficulty, "beta": time_intensity}')
    predict_parser.add_argument('--model-file', '-m', default='models/lnirt_model.pkl',
                                help='Path to trained model file (default: models/lnirt_model.pkl)')
    predict_parser.set_defaults(func=predict_command)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display model statistics')
    stats_parser.add_argument('--model-file', '-m', default='models/lnirt_model.pkl',
                              help='Path to trained model file (default: models/lnirt_model.pkl)')
    stats_parser.add_argument('--top-users', type=int, default=0,
                              help='Show top N users by ability')
    stats_parser.add_argument('--hardest-items', type=int, default=0,
                              help='Show N hardest items')
    stats_parser.set_defaults(func=stats_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
