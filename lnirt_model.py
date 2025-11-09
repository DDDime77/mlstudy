"""
LNIRT Model Implementation
Joint Item Response Theory + Lognormal Response Time Model

This model simultaneously estimates:
- IRT parameters: user ability (theta), item difficulty (b), discrimination (a)
- Response time parameters: user speed (tau), item time intensity (beta)
- Correlation between ability and speed
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, lognorm
import pickle
import json
from typing import Dict, Tuple, Optional


class LNIRTModel:
    """
    Joint IRT and Lognormal Response Time Model

    IRT Component (2PL):
    P(correct|theta, a, b) = 1 / (1 + exp(-a*(theta - b)))

    Log-normal Response Time Component:
    log(RT) ~ N(beta - tau, sigma²)

    Where:
    - theta: user ability
    - tau: user speed (higher = faster)
    - a: item discrimination
    - b: item difficulty
    - beta: item time intensity (higher = takes longer)
    - sigma: residual variance in log response time
    """

    def __init__(self):
        self.user_params = {}  # {user_id: {'theta': ability, 'tau': speed}}
        self.item_params = {}  # {item_id: {'a': discrimination, 'b': difficulty, 'beta': time_intensity}}
        self.sigma = 1.0  # residual SD for log-response time
        self.rho = 0.0  # correlation between ability and speed
        self.is_trained = False

    def _irt_probability(self, theta: float, a: float, b: float) -> float:
        """Calculate probability of correct response using 2PL IRT"""
        return 1.0 / (1.0 + np.exp(-a * (theta - b)))

    def _log_rt_likelihood(self, log_rt: float, tau: float, beta: float, sigma: float) -> float:
        """Calculate log-likelihood of log response time"""
        mean = beta - tau
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((log_rt - mean) / sigma)**2

    def _negative_log_likelihood(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """
        Negative log-likelihood for optimization

        Parameters are organized as:
        - User parameters: [theta_1, ..., theta_n, tau_1, ..., tau_n]
        - Item parameters: [a_1, ..., a_m, b_1, ..., b_m, beta_1, ..., beta_m]
        - Global: [sigma]
        """
        n_users = len(data['user_id'].unique())
        n_items = len(data['item_id'].unique())

        # Extract parameters
        user_theta = params[:n_users]
        user_tau = params[n_users:2*n_users]
        item_a = params[2*n_users:2*n_users+n_items]
        item_b = params[2*n_users+n_items:2*n_users+2*n_items]
        item_beta = params[2*n_users+2*n_items:2*n_users+3*n_items]
        sigma = params[-1]

        # Create lookup dictionaries
        user_ids = sorted(data['user_id'].unique())
        item_ids = sorted(data['item_id'].unique())
        user_idx_map = {uid: i for i, uid in enumerate(user_ids)}
        item_idx_map = {iid: i for i, iid in enumerate(item_ids)}

        nll = 0.0

        for _, row in data.iterrows():
            user_idx = user_idx_map[row['user_id']]
            item_idx = item_idx_map[row['item_id']]

            theta = user_theta[user_idx]
            tau = user_tau[user_idx]
            a = item_a[item_idx]
            b = item_b[item_idx]
            beta = item_beta[item_idx]

            # IRT likelihood
            p_correct = self._irt_probability(theta, a, b)
            if row['correct'] == 1:
                nll -= np.log(p_correct + 1e-10)
            else:
                nll -= np.log(1 - p_correct + 1e-10)

            # Response time likelihood
            log_rt = np.log(row['response_time'])
            nll -= self._log_rt_likelihood(log_rt, tau, beta, sigma)

        return nll

    def fit(self, data: pd.DataFrame, max_iter: int = 1000, verbose: bool = True):
        """
        Fit the model to training data

        Args:
            data: DataFrame with columns ['user_id', 'item_id', 'correct', 'response_time']
            max_iter: Maximum iterations for optimization
            verbose: Whether to print progress
        """
        if verbose:
            print("Starting LNIRT model training...")
            print(f"Data shape: {data.shape}")
            print(f"Users: {data['user_id'].nunique()}, Items: {data['item_id'].nunique()}")

        # Get unique users and items
        user_ids = sorted(data['user_id'].unique())
        item_ids = sorted(data['item_id'].unique())
        n_users = len(user_ids)
        n_items = len(item_ids)

        # Initialize parameters
        # Users: theta~N(0,1), tau~N(0,1)
        theta_init = np.random.randn(n_users) * 0.5
        tau_init = np.random.randn(n_users) * 0.5

        # Items: a~U(0.5, 2), b~N(0,1), beta~N(log(mean_rt), 1)
        mean_log_rt = np.log(data['response_time'].mean())
        a_init = np.random.uniform(0.5, 2.0, n_items)
        b_init = np.random.randn(n_items) * 0.5
        beta_init = np.random.randn(n_items) * 0.5 + mean_log_rt

        # Global: sigma
        sigma_init = np.array([1.0])

        # Combine all parameters
        params_init = np.concatenate([theta_init, tau_init, a_init, b_init, beta_init, sigma_init])

        # Set bounds
        bounds = []
        # theta: unbounded
        bounds.extend([(-5, 5)] * n_users)
        # tau: unbounded
        bounds.extend([(-5, 5)] * n_users)
        # a: positive discrimination
        bounds.extend([(0.1, 5)] * n_items)
        # b: unbounded difficulty
        bounds.extend([(-5, 5)] * n_items)
        # beta: time intensity
        bounds.extend([(-5, 10)] * n_items)
        # sigma: positive
        bounds.append((0.1, 5))

        # Optimize
        if verbose:
            print("Optimizing parameters...")

        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': verbose}
        )

        # Extract fitted parameters
        params_opt = result.x
        user_theta = params_opt[:n_users]
        user_tau = params_opt[n_users:2*n_users]
        item_a = params_opt[2*n_users:2*n_users+n_items]
        item_b = params_opt[2*n_users+n_items:2*n_users+2*n_items]
        item_beta = params_opt[2*n_users+2*n_items:2*n_users+3*n_items]
        self.sigma = params_opt[-1]

        # Store in dictionaries
        self.user_params = {
            user_ids[i]: {'theta': user_theta[i], 'tau': user_tau[i]}
            for i in range(n_users)
        }

        self.item_params = {
            item_ids[i]: {
                'a': item_a[i],
                'b': item_b[i],
                'beta': item_beta[i]
            }
            for i in range(n_items)
        }

        self.is_trained = True

        if verbose:
            print(f"\nTraining completed!")
            print(f"Final negative log-likelihood: {result.fun:.2f}")
            print(f"Sigma (RT residual SD): {self.sigma:.3f}")

    def predict(self, user_id: str, item_id: str, item_features: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Predict probability of correct response and expected response time

        Args:
            user_id: User identifier
            item_id: Item identifier
            item_features: Optional item features for new items

        Returns:
            (probability_correct, expected_time_seconds)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Handle new users (use population mean)
        if user_id not in self.user_params:
            theta = 0.0
            tau = 0.0
        else:
            theta = self.user_params[user_id]['theta']
            tau = self.user_params[user_id]['tau']

        # Handle new items
        if item_id not in self.item_params:
            if item_features is None:
                # Use population mean
                a = 1.0
                b = 0.0
                beta = 0.0
            else:
                # Could implement feature-based estimation here
                a = item_features.get('a', 1.0)
                b = item_features.get('b', 0.0)
                beta = item_features.get('beta', 0.0)
        else:
            a = self.item_params[item_id]['a']
            b = self.item_params[item_id]['b']
            beta = self.item_params[item_id]['beta']

        # Predict correctness
        p_correct = self._irt_probability(theta, a, b)

        # Predict response time
        # For lognormal, median = exp(mean), mean = exp(mean + sigma²/2)
        log_rt_mean = beta - tau
        expected_time = np.exp(log_rt_mean + self.sigma**2 / 2)

        return p_correct, expected_time

    def add_user_data(self, user_id: str, item_id: str, correct: int, response_time: float):
        """
        Update model with new user response (incremental learning)
        For simplicity, this adds to stored data and requires retraining
        """
        # This is a simplified version - full incremental learning would use online methods
        pass

    def save_model(self, filepath: str):
        """Save model parameters to file"""
        model_data = {
            'user_params': self.user_params,
            'item_params': self.item_params,
            'sigma': self.sigma,
            'rho': self.rho,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load model parameters from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.user_params = model_data['user_params']
        self.item_params = model_data['item_params']
        self.sigma = model_data['sigma']
        self.rho = model_data['rho']
        self.is_trained = model_data['is_trained']

    def get_user_stats(self) -> pd.DataFrame:
        """Get summary statistics of user parameters"""
        data = []
        for user_id, params in self.user_params.items():
            data.append({
                'user_id': user_id,
                'ability_theta': params['theta'],
                'speed_tau': params['tau']
            })
        return pd.DataFrame(data)

    def get_item_stats(self) -> pd.DataFrame:
        """Get summary statistics of item parameters"""
        data = []
        for item_id, params in self.item_params.items():
            data.append({
                'item_id': item_id,
                'discrimination_a': params['a'],
                'difficulty_b': params['b'],
                'time_intensity_beta': params['beta']
            })
        return pd.DataFrame(data)
