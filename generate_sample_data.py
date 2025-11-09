"""
Generate sample datasets for LNIRT model training

This script creates realistic sample data for educational/testing purposes
"""

import numpy as np
import pandas as pd
import json

np.random.seed(42)


def generate_sample_data(
    n_users: int = 50,
    n_items: int = 100,
    responses_per_user: int = 30,
    output_file: str = 'data/sample_training_data.csv'
):
    """
    Generate synthetic training data

    Data structure:
    - user_id: Unique identifier for each student/user
    - item_id: Unique identifier for each question/task
    - correct: Binary outcome (1=correct, 0=incorrect)
    - response_time: Time taken to answer in seconds

    The generation process:
    1. Assign each user a latent ability (theta) and speed (tau)
    2. Assign each item difficulty (b), discrimination (a), and time intensity (beta)
    3. Generate responses based on IRT model
    4. Generate response times based on lognormal model
    """

    # Generate user parameters
    # theta: ability ~ N(0, 1) - higher means more capable
    # tau: speed ~ N(0, 0.5) - higher means faster
    user_theta = np.random.randn(n_users)
    user_tau = np.random.randn(n_users) * 0.5

    # Generate item parameters
    # a: discrimination ~ Uniform(0.5, 2.5) - how well item distinguishes ability
    # b: difficulty ~ N(0, 1) - higher means harder
    # beta: time intensity ~ N(4, 0.8) - base log-time for item
    item_a = np.random.uniform(0.5, 2.5, n_items)
    item_b = np.random.randn(n_items)
    item_beta = np.random.randn(n_items) * 0.8 + 4  # mean ~54 seconds

    # Generate responses
    data = []

    for user_id in range(n_users):
        # Each user answers a random subset of items
        items_answered = np.random.choice(n_items, size=responses_per_user, replace=False)

        for item_id in items_answered:
            theta = user_theta[user_id]
            tau = user_tau[user_id]
            a = item_a[item_id]
            b = item_b[item_id]
            beta = item_beta[item_id]

            # Generate correctness using IRT
            p_correct = 1.0 / (1.0 + np.exp(-a * (theta - b)))
            correct = 1 if np.random.rand() < p_correct else 0

            # Generate response time using lognormal
            # log(RT) ~ N(beta - tau, sigma=0.5)
            log_rt = np.random.normal(beta - tau, 0.5)
            response_time = np.exp(log_rt)

            data.append({
                'user_id': f'user_{user_id:03d}',
                'item_id': f'item_{item_id:03d}',
                'correct': correct,
                'response_time': response_time
            })

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} training samples")
    print(f"Saved to: {output_file}")
    print(f"\nSummary statistics:")
    print(f"  Accuracy: {df['correct'].mean():.2%}")
    print(f"  Mean response time: {df['response_time'].mean():.1f} seconds")
    print(f"  Median response time: {df['response_time'].median():.1f} seconds")

    return df


def generate_item_bank(
    n_items: int = 100,
    output_file: str = 'data/item_bank.json'
):
    """
    Generate a bank of items (questions/tasks) with metadata

    Each item includes:
    - item_id: Unique identifier
    - question_text: The actual question
    - topic: Subject area
    - estimated_difficulty: Rough difficulty estimate (1-5)
    """

    topics = ['algebra', 'geometry', 'calculus', 'statistics', 'trigonometry']

    items = []
    for i in range(n_items):
        topic = np.random.choice(topics)
        difficulty = np.random.randint(1, 6)

        # Generate sample question text
        question_templates = [
            f"Solve the {topic} problem: equation {i}",
            f"Calculate the {topic} expression for case {i}",
            f"Find the solution to {topic} question {i}",
            f"Determine the {topic} value in problem {i}",
        ]

        items.append({
            'item_id': f'item_{i:03d}',
            'question_text': np.random.choice(question_templates),
            'topic': topic,
            'estimated_difficulty': difficulty
        })

    with open(output_file, 'w') as f:
        json.dump(items, f, indent=2)

    print(f"\nGenerated item bank with {len(items)} items")
    print(f"Saved to: {output_file}")
    print(f"Topics: {', '.join(set(item['topic'] for item in items))}")


def create_user_data_template(output_file: str = 'data/user_data_template.csv'):
    """
    Create a template file showing the format for user training data

    This template demonstrates the required format for users who want to
    add their own training data to the model.
    """

    template_data = [
        {
            'user_id': 'user_001',
            'item_id': 'item_001',
            'correct': 1,
            'response_time': 45.2,
            'notes': 'User got this correct in 45.2 seconds'
        },
        {
            'user_id': 'user_001',
            'item_id': 'item_002',
            'correct': 0,
            'response_time': 120.5,
            'notes': 'User got this wrong, took 2+ minutes'
        },
        {
            'user_id': 'user_002',
            'item_id': 'item_001',
            'correct': 1,
            'response_time': 30.8,
            'notes': 'Faster user, correct answer'
        },
    ]

    df = pd.DataFrame(template_data)
    df.to_csv(output_file, index=False)

    print(f"\nCreated user data template: {output_file}")
    print("Use this format when adding your own training data")
    print("\nRequired columns:")
    print("  - user_id: string identifier for the user")
    print("  - item_id: string identifier for the question/task")
    print("  - correct: 1 for correct, 0 for incorrect")
    print("  - response_time: time in seconds (can be decimal)")
    print("  - notes: (optional) any comments about this response")


if __name__ == '__main__':
    print("=" * 60)
    print("SAMPLE DATA GENERATOR FOR LNIRT MODEL")
    print("=" * 60)

    # Generate training data
    print("\n1. Generating training data...")
    generate_sample_data(
        n_users=50,
        n_items=100,
        responses_per_user=30,
        output_file='data/sample_training_data.csv'
    )

    # Generate item bank
    print("\n2. Generating item bank...")
    generate_item_bank(
        n_items=100,
        output_file='data/item_bank.json'
    )

    # Create template
    print("\n3. Creating user data template...")
    create_user_data_template(
        output_file='data/user_data_template.csv'
    )

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - data/sample_training_data.csv: Training data (1500 responses)")
    print("  - data/item_bank.json: Question bank with metadata")
    print("  - data/user_data_template.csv: Template for your own data")
