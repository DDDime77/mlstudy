"""
Generate topic-specific training data for LNIRT model

Generates data for:
- Mathematics: algebra, geometry, calculus, statistics, trigonometry
- Economics: microeconomics, macroeconomics, global_economics
"""

import numpy as np
import pandas as pd
import json

np.random.seed(42)


def generate_topic_training_data(
    n_users: int = 60,
    topics_config: dict = None,
    responses_per_user_per_topic: int = 10,
    output_file: str = 'data/topic_training_data.csv'
):
    """
    Generate synthetic training data with topics

    Args:
        n_users: Number of users
        topics_config: Dictionary of topics with their base difficulty
        responses_per_user_per_topic: How many questions per topic per user
        output_file: Where to save the data
    """

    if topics_config is None:
        topics_config = {
            # Mathematics topics
            'algebra': {'base_difficulty': -0.3, 'base_time': 3.8},
            'geometry': {'base_difficulty': 0.0, 'base_time': 4.2},
            'calculus': {'base_difficulty': 0.8, 'base_time': 4.8},
            'statistics': {'base_difficulty': 0.3, 'base_time': 4.5},
            'trigonometry': {'base_difficulty': 0.5, 'base_time': 4.0},

            # Economics topics
            'microeconomics': {'base_difficulty': 0.2, 'base_time': 4.3},
            'macroeconomics': {'base_difficulty': 0.4, 'base_time': 4.6},
            'global_economics': {'base_difficulty': 0.7, 'base_time': 5.0},
        }

    # Generate user parameters
    # theta: ability ~ N(0, 1)
    # tau: speed ~ N(0, 0.5)
    user_theta = np.random.randn(n_users)
    user_tau = np.random.randn(n_users) * 0.5

    # Generate items for each topic
    item_counter = 0
    all_items = []

    for topic, config in topics_config.items():
        n_items_for_topic = responses_per_user_per_topic * 3  # More items than responses
        base_diff = config['base_difficulty']
        base_time = config['base_time']

        for _ in range(n_items_for_topic):
            item_id = f"{topic}_{item_counter:03d}"

            # Discrimination: ~Uniform(0.5, 2.5)
            a = np.random.uniform(0.5, 2.5)

            # Difficulty: centered around topic base_difficulty
            b = np.random.randn() * 0.8 + base_diff

            # Time intensity: centered around topic base_time
            beta = np.random.randn() * 0.6 + base_time

            all_items.append({
                'item_id': item_id,
                'topic': topic,
                'a': a,
                'b': b,
                'beta': beta
            })
            item_counter += 1

    # Generate responses
    data = []

    for user_id in range(n_users):
        theta = user_theta[user_id]
        tau = user_tau[user_id]

        # Each user answers questions from each topic
        for topic in topics_config.keys():
            topic_items = [item for item in all_items if item['topic'] == topic]
            selected_items = np.random.choice(len(topic_items),
                                            size=min(responses_per_user_per_topic, len(topic_items)),
                                            replace=False)

            for item_idx in selected_items:
                item = topic_items[item_idx]

                a = item['a']
                b = item['b']
                beta = item['beta']

                # Generate correctness using IRT
                p_correct = 1.0 / (1.0 + np.exp(-a * (theta - b)))
                correct = 1 if np.random.rand() < p_correct else 0

                # Generate response time using lognormal
                log_rt = np.random.normal(beta - tau, 0.5)
                response_time = np.exp(log_rt)

                data.append({
                    'user_id': f'user_{user_id:03d}',
                    'item_id': item['item_id'],
                    'topic': topic,
                    'correct': correct,
                    'response_time': response_time
                })

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Generated {len(df)} training samples")
    print(f"Saved to: {output_file}")
    print(f"\nTopic breakdown:")
    for topic in df['topic'].unique():
        topic_data = df[df['topic'] == topic]
        print(f"  {topic:20s}: {len(topic_data):4d} responses, "
              f"accuracy={topic_data['correct'].mean():.1%}, "
              f"mean_time={topic_data['response_time'].mean():.1f}s")

    return df


def generate_topic_item_bank(topics_config: dict = None, output_file: str = 'data/topic_item_bank.json'):
    """Generate item bank with actual questions for each topic"""

    if topics_config is None:
        topics_config = {
            'algebra': {'count': 30},
            'geometry': {'count': 20},
            'calculus': {'count': 20},
            'statistics': {'count': 15},
            'trigonometry': {'count': 15},
            'microeconomics': {'count': 30},
            'macroeconomics': {'count': 25},
            'global_economics': {'count': 25},
        }

    # Real questions for each topic
    questions_db = {
        'algebra': [
            "Solve for x: 2x + 5 = 15",
            "Simplify: (x² - 4)/(x - 2)",
            "Factor: x² + 7x + 12",
            "Solve the system: x + y = 10, 2x - y = 5",
            "Expand: (x + 3)(x - 4)",
            "Solve: 3(x - 2) = 2x + 4",
            "Find the value of x: x/4 + 3 = 7",
            "Simplify: 2x + 3x - 5x + 8",
            "Solve for y: 4y - 7 = 2y + 9",
            "Factor completely: 2x² + 8x + 6",
        ],
        'geometry': [
            "Find the area of a circle with radius 5cm",
            "Calculate the perimeter of a rectangle with length 8cm and width 5cm",
            "Find the volume of a cube with side length 4cm",
            "What is the sum of interior angles in a pentagon?",
            "Calculate the hypotenuse of a right triangle with sides 3 and 4",
            "Find the area of a triangle with base 10cm and height 6cm",
            "Calculate the circumference of a circle with diameter 14cm",
            "Find the surface area of a sphere with radius 3cm",
        ],
        'calculus': [
            "Find the derivative of f(x) = x³ + 2x²",
            "Integrate: ∫(3x² + 2x)dx",
            "Find dy/dx for y = sin(x) + cos(x)",
            "Calculate the limit: lim(x→2) (x² - 4)/(x - 2)",
            "Find the second derivative of f(x) = e^x",
            "Evaluate: ∫₀¹ x² dx",
            "Find the critical points of f(x) = x³ - 3x + 2",
            "Calculate d/dx[ln(x)]",
        ],
        'statistics': [
            "Calculate the mean of: 4, 8, 6, 5, 7",
            "Find the median of: 12, 8, 15, 10, 9",
            "Calculate the standard deviation of: 2, 4, 4, 4, 5, 5, 7, 9",
            "What is the probability of rolling a 6 on a fair die?",
            "Find the mode of: 3, 7, 3, 2, 5, 3, 8",
            "Calculate the range of: 15, 23, 12, 19, 27, 11",
        ],
        'trigonometry': [
            "Find sin(30°)",
            "Calculate cos(60°)",
            "What is tan(45°)?",
            "Solve: sin(x) = 0.5 for 0° ≤ x ≤ 360°",
            "Convert 90° to radians",
            "Find the value of sin²(x) + cos²(x)",
            "Calculate: 2sin(30°)cos(30°)",
        ],
        'microeconomics': [
            "Define opportunity cost and provide an example",
            "Explain the law of demand",
            "What is price elasticity of demand?",
            "Describe the concept of marginal utility",
            "What is a perfectly competitive market?",
            "Explain the difference between accounting profit and economic profit",
            "What is consumer surplus?",
            "Define price discrimination",
            "Explain the concept of diminishing marginal returns",
            "What is the equilibrium price in a market?",
        ],
        'macroeconomics': [
            "Define GDP and its components",
            "Explain the difference between nominal and real GDP",
            "What is inflation and how is it measured?",
            "Describe the role of the central bank",
            "What is monetary policy?",
            "Explain fiscal policy and its tools",
            "What is the unemployment rate?",
            "Describe the business cycle phases",
            "What is aggregate demand?",
            "Explain the multiplier effect",
        ],
        'global_economics': [
            "What is comparative advantage in international trade?",
            "Explain the balance of payments",
            "What are exchange rates and how are they determined?",
            "Describe the impact of tariffs on trade",
            "What is globalization?",
            "Explain the role of the World Trade Organization",
            "What is foreign direct investment (FDI)?",
            "Describe the effects of trade deficits",
            "What are emerging markets?",
            "Explain currency appreciation and depreciation",
        ]
    }

    items = []
    item_counter = 0

    for topic, config in topics_config.items():
        count = config['count']
        topic_questions = questions_db.get(topic, [])

        for i in range(count):
            # Cycle through available questions
            question_text = topic_questions[i % len(topic_questions)] if topic_questions else f"Question {i+1} about {topic}"

            items.append({
                'item_id': f'{topic}_{item_counter:03d}',
                'topic': topic,
                'question_text': question_text,
                'difficulty_level': np.random.randint(1, 6)
            })
            item_counter += 1

    with open(output_file, 'w') as f:
        json.dump(items, f, indent=2)

    print(f"\nGenerated item bank with {len(items)} items")
    print(f"Saved to: {output_file}")
    print(f"Topics: {', '.join(topics_config.keys())}")

    return items


if __name__ == '__main__':
    print("=" * 70)
    print("TOPIC-SPECIFIC DATA GENERATOR FOR LNIRT MODEL")
    print("=" * 70)

    # Generate training data
    print("\n1. Generating topic-specific training data...")
    df = generate_topic_training_data(
        n_users=60,
        responses_per_user_per_topic=10,
        output_file='data/topic_training_data.csv'
    )

    # Generate item bank
    print("\n2. Generating topic-specific item bank...")
    items = generate_topic_item_bank(
        output_file='data/topic_item_bank.json'
    )

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE!")
    print("=" * 70)
