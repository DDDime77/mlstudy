"""
Topic-Based LNIRT Model
Simplified model where each topic is completely independent
Uses difficulty levels (1-3) instead of individual items
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
import os
from typing import Dict, Tuple, Optional


class TopicLNIRTModel:
    """
    Simplified LNIRT model for a single topic

    - Users have: ability (theta), speed (tau) for THIS topic only
    - Difficulty levels (1-3) have: discrimination (a), difficulty (b), time intensity (beta)
    - Model is trained only on data for this specific topic
    """

    def __init__(self, topic: str):
        self.topic = topic
        self.user_params = {}  # {user_id: {'theta': ability, 'tau': speed}}
        self.difficulty_params = {
            1: {'a': 1.0, 'b': -0.5, 'beta': 3.5},  # Easy
            2: {'a': 1.2, 'b': 0.0, 'beta': 4.0},   # Medium
            3: {'a': 1.5, 'b': 0.8, 'beta': 4.5}    # Hard
        }
        self.sigma = 0.5  # response time variance
        self.is_trained = False

    def _irt_probability(self, theta: float, a: float, b: float) -> float:
        """Calculate probability of correct response using 2PL IRT"""
        return 1.0 / (1.0 + np.exp(-a * (theta - b)))

    def _log_rt_likelihood(self, log_rt: float, tau: float, beta: float, sigma: float) -> float:
        """Calculate log-likelihood of log response time"""
        mean = beta - tau
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((log_rt - mean) / sigma)**2

    def _joint_log_likelihood(self, params, data_subset, user_idx_map, param_type):
        """
        Calculate joint log-likelihood for LNIRT model

        Args:
            params: Parameters to optimize
            data_subset: Data for this optimization
            user_idx_map: Mapping from user_id to index
            param_type: 'user' or 'difficulty'
        """
        log_likelihood = 0.0

        for _, row in data_subset.iterrows():
            user_id = row['user_id']
            difficulty = int(row['difficulty'])
            correct = row['correct']
            response_time = row['response_time']

            # Get parameters
            if param_type == 'user':
                theta = params[user_idx_map[user_id] * 2]
                tau = params[user_idx_map[user_id] * 2 + 1]
                a = self.difficulty_params[difficulty]['a']
                b = self.difficulty_params[difficulty]['b']
                beta = self.difficulty_params[difficulty]['beta']
            else:  # param_type == 'difficulty'
                theta = self.user_params[user_id]['theta']
                tau = self.user_params[user_id]['tau']
                # params = [a, b, beta] for this difficulty
                a = params[0]
                b = params[1]
                beta = params[2]

            # IRT component: P(correct | theta, a, b)
            p_correct = self._irt_probability(theta, a, b)
            p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)  # Numerical stability

            if correct == 1:
                log_likelihood += np.log(p_correct)
            else:
                log_likelihood += np.log(1 - p_correct)

            # Lognormal RT component: P(log(RT) | tau, beta, sigma)
            log_rt = np.log(response_time + 0.1)  # Add small constant to avoid log(0)
            log_likelihood += self._log_rt_likelihood(log_rt, tau, beta, self.sigma)

        return -log_likelihood  # Return negative for minimization

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """
        Train model on topic-specific data using LNIRT maximum likelihood estimation

        Args:
            data: DataFrame with ['user_id', 'difficulty', 'correct', 'response_time']
            verbose: Print training progress
        """
        if verbose:
            print(f"Training {self.topic} model...")
            print(f"  Data: {len(data)} responses")
            print(f"  Users: {data['user_id'].nunique()}")
            print(f"  Difficulties: {sorted(data['difficulty'].unique())}")

        # Get unique users
        user_ids = sorted(data['user_id'].unique())
        n_users = len(user_ids)

        # Initialize user parameters
        user_theta = np.random.randn(n_users) * 0.3
        user_tau = np.random.randn(n_users) * 0.3

        # Initialize user_params dict for use in optimization
        self.user_params = {
            user_ids[i]: {'theta': float(user_theta[i]), 'tau': float(user_tau[i])}
            for i in range(n_users)
        }

        # EM-like algorithm: alternate between optimizing user and difficulty parameters
        for iteration in range(5):  # EM iterations (reduced for efficiency)
            # Step 1: Update difficulty parameters (holding user parameters fixed)
            for diff_level in [1, 2, 3]:
                diff_data = data[data['difficulty'] == diff_level]
                if len(diff_data) > 0:
                    # Get initial parameters
                    initial_params = [
                        self.difficulty_params[diff_level]['a'],
                        self.difficulty_params[diff_level]['b'],
                        self.difficulty_params[diff_level]['beta']
                    ]

                    # Optimize difficulty parameters
                    result = minimize(
                        self._joint_log_likelihood,
                        initial_params,
                        args=(diff_data, None, 'difficulty'),
                        method='L-BFGS-B',
                        bounds=[(0.5, 3.0), (-3.0, 3.0), (2.0, 6.0)],  # Reasonable bounds
                        options={'maxiter': 50}  # Limit iterations
                    )

                    if result.success:
                        self.difficulty_params[diff_level]['a'] = float(result.x[0])
                        self.difficulty_params[diff_level]['b'] = float(result.x[1])
                        self.difficulty_params[diff_level]['beta'] = float(result.x[2])

            # Step 2: Update user parameters INDIVIDUALLY (much faster than joint optimization)
            for user_id in user_ids:
                user_data = data[data['user_id'] == user_id]
                if len(user_data) > 0:
                    # Define single-user likelihood
                    def single_user_likelihood(params):
                        theta, tau = params
                        log_like = 0.0
                        for _, row in user_data.iterrows():
                            difficulty = int(row['difficulty'])
                            correct = row['correct']
                            response_time = row['response_time']

                            a = self.difficulty_params[difficulty]['a']
                            b = self.difficulty_params[difficulty]['b']
                            beta = self.difficulty_params[difficulty]['beta']

                            # IRT component
                            p_correct = self._irt_probability(theta, a, b)
                            p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
                            log_like += np.log(p_correct) if correct == 1 else np.log(1 - p_correct)

                            # RT component
                            log_rt = np.log(response_time + 0.1)
                            log_like += self._log_rt_likelihood(log_rt, tau, beta, self.sigma)

                        return -log_like

                    # Optimize this user's parameters
                    initial_params = [
                        self.user_params[user_id]['theta'],
                        self.user_params[user_id]['tau']
                    ]

                    result = minimize(
                        single_user_likelihood,
                        initial_params,
                        method='L-BFGS-B',
                        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
                        options={'maxiter': 50}
                    )

                    if result.success:
                        self.user_params[user_id]['theta'] = float(result.x[0])
                        self.user_params[user_id]['tau'] = float(result.x[1])

            if verbose and iteration % 2 == 0:
                print(f"  Iteration {iteration + 1}/5...")

        self.is_trained = True

        if verbose:
            print(f"  ✓ Training complete (LNIRT ML estimation)")
            print(f"  Users trained: {len(self.user_params)}")

    def fit_user_specific(self, user_data: pd.DataFrame, user_id: str, verbose: bool = False):
        """
        Train user-specific parameters using their prediction history with LNIRT ML estimation

        This uses data from the predictions table which includes both predicted
        and actual results to refine the user's ability and speed parameters.

        Args:
            user_data: DataFrame with columns ['difficulty', 'correct', 'response_time',
                                               'predicted_correct', 'predicted_time']
            user_id: User identifier
            verbose: Print progress
        """
        if verbose:
            print(f"  Training user-specific parameters for {user_id}...")
            print(f"  Using {len(user_data)} completed tasks")

        # Initialize user if not exists
        if user_id not in self.user_params:
            if self.user_params:
                # Use population average as starting point
                avg_theta = np.mean([p['theta'] for p in self.user_params.values()])
                avg_tau = np.mean([p['tau'] for p in self.user_params.values()])
            else:
                avg_theta = 0.0
                avg_tau = 0.0
            self.user_params[user_id] = {'theta': avg_theta, 'tau': avg_tau}

        # Define likelihood function for single user
        def user_log_likelihood(params, data):
            theta, tau = params
            log_like = 0.0

            for _, row in data.iterrows():
                difficulty = int(row['difficulty'])
                correct = row['correct']
                response_time = row['response_time']

                # Get difficulty parameters
                a = self.difficulty_params[difficulty]['a']
                b = self.difficulty_params[difficulty]['b']
                beta = self.difficulty_params[difficulty]['beta']

                # IRT component
                p_correct = self._irt_probability(theta, a, b)
                p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)

                if correct == 1:
                    log_like += np.log(p_correct)
                else:
                    log_like += np.log(1 - p_correct)

                # Lognormal RT component
                log_rt = np.log(response_time + 0.1)
                log_like += self._log_rt_likelihood(log_rt, tau, beta, self.sigma)

            return -log_like  # Negative for minimization

        # Optimize user-specific parameters
        initial_params = [
            self.user_params[user_id]['theta'],
            self.user_params[user_id]['tau']
        ]

        result = minimize(
            user_log_likelihood,
            initial_params,
            args=(user_data,),
            method='L-BFGS-B',
            bounds=[(-3.0, 3.0), (-3.0, 3.0)]
        )

        if result.success:
            self.user_params[user_id]['theta'] = float(result.x[0])
            self.user_params[user_id]['tau'] = float(result.x[1])
        else:
            # Fallback to simple estimation if optimization fails
            if verbose:
                print(f"  Note: Using fallback estimation (optimization didn't converge)")

            overall_accuracy = user_data['correct'].mean()
            overall_time = user_data['response_time'].mean()

            if 0.05 < overall_accuracy < 0.95:
                theta_estimate = np.log(overall_accuracy / (1 - overall_accuracy))
                self.user_params[user_id]['theta'] = 0.7 * self.user_params[user_id]['theta'] + 0.3 * theta_estimate

            beta_ref = self.difficulty_params[2]['beta']
            tau_estimate = beta_ref - np.log(overall_time + 1)
            self.user_params[user_id]['tau'] = 0.7 * self.user_params[user_id]['tau'] + 0.3 * tau_estimate

        if verbose:
            print(f"  ✓ User parameters updated (LNIRT ML estimation)")
            print(f"    Ability (θ): {self.user_params[user_id]['theta']:.3f}")
            print(f"    Speed (τ): {self.user_params[user_id]['tau']:.3f}")

    def predict(self, user_id: str, difficulty: int) -> Tuple[float, float]:
        """
        Predict correctness and time for user on difficulty level

        Args:
            user_id: User identifier
            difficulty: Difficulty level (1, 2, or 3)

        Returns:
            (probability_correct, expected_time_seconds)
        """
        if difficulty not in [1, 2, 3]:
            raise ValueError(f"Difficulty must be 1, 2, or 3, got {difficulty}")

        # Get user parameters (or use defaults for new users)
        if user_id in self.user_params:
            theta = self.user_params[user_id]['theta']
            tau = self.user_params[user_id]['tau']
        else:
            # New user - use population average
            if self.user_params:
                theta = np.mean([p['theta'] for p in self.user_params.values()])
                tau = np.mean([p['tau'] for p in self.user_params.values()])
            else:
                theta = 0.0
                tau = 0.0

        # Get difficulty parameters
        a = self.difficulty_params[difficulty]['a']
        b = self.difficulty_params[difficulty]['b']
        beta = self.difficulty_params[difficulty]['beta']

        # Predict correctness
        p_correct = self._irt_probability(theta, a, b)

        # Predict time
        log_rt_mean = beta - tau
        expected_time = np.exp(log_rt_mean)

        return float(p_correct), float(expected_time)

    def update_from_response(self, user_id: str, difficulty: int, correct: int, response_time: float):
        """
        Update model parameters based on actual user response
        Implements personalized learning

        Args:
            user_id: User identifier
            difficulty: Difficulty level (1, 2, or 3)
            correct: 1 if correct, 0 if incorrect
            response_time: Actual time taken in seconds
        """
        # Initialize user if new
        if user_id not in self.user_params:
            if self.user_params:
                # Use population average as starting point
                avg_theta = np.mean([p['theta'] for p in self.user_params.values()])
                avg_tau = np.mean([p['tau'] for p in self.user_params.values()])
            else:
                avg_theta = 0.0
                avg_tau = 0.0

            self.user_params[user_id] = {'theta': avg_theta, 'tau': avg_tau}

        # Get current parameters
        current_theta = self.user_params[user_id]['theta']
        current_tau = self.user_params[user_id]['tau']

        # Update using exponential moving average (quick adaptation)
        learning_rate = 0.3

        # Update ability based on correctness
        expected_correct = self._irt_probability(
            current_theta,
            self.difficulty_params[difficulty]['a'],
            self.difficulty_params[difficulty]['b']
        )
        error = correct - expected_correct
        new_theta = current_theta + learning_rate * error

        # Update speed based on response time
        expected_log_time = self.difficulty_params[difficulty]['beta'] - current_tau
        actual_log_time = np.log(response_time + 0.1)
        time_error = expected_log_time - actual_log_time
        new_tau = current_tau + learning_rate * time_error

        # Store updated parameters
        self.user_params[user_id]['theta'] = float(new_theta)
        self.user_params[user_id]['tau'] = float(new_tau)

    def save(self, filepath: str):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'topic': self.topic,
                'user_params': self.user_params,
                'difficulty_params': self.difficulty_params,
                'sigma': self.sigma,
                'is_trained': self.is_trained
            }, f)

    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.topic = data['topic']
        self.user_params = data['user_params']
        self.difficulty_params = data['difficulty_params']
        self.sigma = data['sigma']
        self.is_trained = data['is_trained']

    def get_stats(self) -> Dict:
        """Get model statistics"""
        stats = {
            'topic': self.topic,
            'n_users': len(self.user_params),
            'difficulty_params': self.difficulty_params
        }

        if self.user_params:
            thetas = [p['theta'] for p in self.user_params.values()]
            taus = [p['tau'] for p in self.user_params.values()]
            stats['user_ability'] = {
                'mean': float(np.mean(thetas)),
                'std': float(np.std(thetas)),
                'min': float(np.min(thetas)),
                'max': float(np.max(thetas))
            }
            stats['user_speed'] = {
                'mean': float(np.mean(taus)),
                'std': float(np.std(taus)),
                'min': float(np.min(taus)),
                'max': float(np.max(taus))
            }

        return stats


class TopicModelManager:
    """Manages multiple topic-specific models"""

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        os.makedirs(models_dir, exist_ok=True)

    def get_model(self, topic: str) -> TopicLNIRTModel:
        """Get or create model for topic"""
        if topic not in self.models:
            model = TopicLNIRTModel(topic)
            model_path = os.path.join(self.models_dir, f"{topic}.pkl")
            if os.path.exists(model_path):
                model.load(model_path)
            self.models[topic] = model
        return self.models[topic]

    def save_model(self, topic: str):
        """Save model for topic"""
        if topic in self.models:
            model_path = os.path.join(self.models_dir, f"{topic}.pkl")
            self.models[topic].save(model_path)

    def list_topics(self) -> list:
        """List all available topics"""
        topics = []
        if os.path.exists(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith('.pkl'):
                    topics.append(f[:-4])
        return sorted(topics)
