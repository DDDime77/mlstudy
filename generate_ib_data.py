"""
Generate IB Diploma Programme training data
Subjects: Math AA, Economics
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def generate_ib_training_data(
    n_users: int = 50,
    responses_per_user_per_difficulty: int = 10,
    output_dir: str = 'data/ib'
):
    """
    Generate IB DP training data

    Topics:
    - Math AA: numbers, algebra, functions, geometry, trigonometry, calculus, statistics
    - Economics: microeconomics, macroeconomics, global_economics
    """

    os.makedirs(output_dir, exist_ok=True)

    # Define topics with their base characteristics
    topics_config = {
        # Math AA topics
        'numbers': {'base_accuracy': 0.65, 'base_time': 45},
        'algebra': {'base_accuracy': 0.60, 'base_time': 60},
        'functions': {'base_accuracy': 0.50, 'base_time': 80},
        'geometry': {'base_accuracy': 0.55, 'base_time': 70},
        'trigonometry': {'base_accuracy': 0.52, 'base_time': 75},
        'calculus': {'base_accuracy': 0.40, 'base_time': 120},
        'statistics': {'base_accuracy': 0.58, 'base_time': 90},

        # Economics topics
        'microeconomics': {'base_accuracy': 0.55, 'base_time': 100},
        'macroeconomics': {'base_accuracy': 0.50, 'base_time': 110},
        'global_economics': {'base_accuracy': 0.45, 'base_time': 130}
    }

    # Generate user abilities (vary per user, same across topics for realism)
    user_abilities = np.random.randn(n_users) * 0.5  # -1.5 to +1.5 roughly
    user_speeds = np.random.randn(n_users) * 0.3  # speed variation

    for topic, config in topics_config.items():
        print(f"Generating data for {topic}...")

        data = []
        base_accuracy = config['base_accuracy']
        base_time = config['base_time']

        for user_id in range(n_users):
            user_ability = user_abilities[user_id]
            user_speed = user_speeds[user_id]

            for difficulty in [1, 2, 3]:
                # Generate multiple responses per difficulty
                for _ in range(responses_per_user_per_difficulty):

                    # Difficulty affects accuracy and time
                    # Difficulty 1: easier (higher accuracy, less time)
                    # Difficulty 3: harder (lower accuracy, more time)
                    difficulty_multiplier = 1.0 + (difficulty - 2) * 0.3

                    # Calculate probability of correct
                    # Higher ability = higher accuracy
                    # Higher difficulty = lower accuracy
                    p_correct = base_accuracy * (1 + user_ability * 0.3) / difficulty_multiplier
                    p_correct = np.clip(p_correct, 0.05, 0.95)

                    # Generate correctness
                    correct = 1 if np.random.rand() < p_correct else 0

                    # Calculate response time
                    # Higher difficulty = more time
                    # Faster user (higher user_speed) = less time
                    expected_time = base_time * difficulty_multiplier / (1 + user_speed * 0.3)
                    # Add lognormal noise
                    response_time = np.random.lognormal(np.log(expected_time), 0.3)
                    response_time = max(5.0, response_time)  # Minimum 5 seconds

                    data.append({
                        'user_id': f'user_{user_id:03d}',
                        'difficulty': difficulty,
                        'correct': correct,
                        'response_time': response_time
                    })

        # Save to CSV
        df = pd.DataFrame(data)
        output_file = os.path.join(output_dir, f'{topic}.csv')
        df.to_csv(output_file, index=False)

        print(f"  ✓ Saved {len(df)} samples to {output_file}")
        print(f"    Accuracy: {df['correct'].mean():.1%}")
        print(f"    Mean time: {df['response_time'].mean():.1f}s")

        # Print stats by difficulty
        for diff in [1, 2, 3]:
            diff_data = df[df['difficulty'] == diff]
            print(f"    Difficulty {diff}: accuracy={diff_data['correct'].mean():.1%}, "
                  f"time={diff_data['response_time'].mean():.1f}s")

    print(f"\n✓ Generated training data for {len(topics_config)} topics")
    print(f"  Output directory: {output_dir}/")


if __name__ == '__main__':
    print("=" * 70)
    print("IB DIPLOMA PROGRAMME TRAINING DATA GENERATOR")
    print("=" * 70)
    print()

    generate_ib_training_data(
        n_users=50,
        responses_per_user_per_difficulty=10,
        output_dir='data/ib'
    )

    print()
    print("=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print("Topics generated:")
    print("  Math AA: numbers, algebra, functions, geometry, trigonometry, calculus, statistics")
    print("  Economics: microeconomics, macroeconomics, global_economics")
    print()
    print("Each topic has:")
    print("  - 50 users")
    print("  - 3 difficulty levels (1=easy, 2=medium, 3=hard)")
    print("  - 10 responses per user per difficulty")
    print("  - Total: 1500 responses per topic")
