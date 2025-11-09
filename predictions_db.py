"""
Predictions Database Management
Tracks predictions and actual results for model updates
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List


class PredictionsDB:
    """SQLite database for tracking predictions and actual results"""

    def __init__(self, db_path: str = 'predictions.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty INTEGER NOT NULL,
            predicted_correct REAL NOT NULL,
            predicted_time REAL NOT NULL,
            actual_correct INTEGER,
            actual_time REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        ''')

        # Training data table (aggregated from actual results)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty INTEGER NOT NULL,
            correct INTEGER NOT NULL,
            response_time REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')

        # Indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_topic ON predictions(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_topic ON training_data(topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_user ON training_data(user_id)')

        self.conn.commit()

    def add_prediction(self, user_id: str, topic: str, difficulty: int,
                      predicted_correct: float, predicted_time: float) -> int:
        """
        Add a new prediction

        Returns:
            task_id of the created prediction
        """
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO predictions (user_id, topic, difficulty, predicted_correct, predicted_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, topic, difficulty, predicted_correct, predicted_time, datetime.now().isoformat()))

        self.conn.commit()
        return cursor.lastrowid

    def update_prediction(self, task_id: int, actual_correct: int, actual_time: float):
        """
        Update prediction with actual results

        Args:
            task_id: The task ID to update
            actual_correct: 1 if correct, 0 if incorrect
            actual_time: Actual time taken in seconds
        """
        cursor = self.conn.cursor()

        # Get the prediction
        cursor.execute('SELECT user_id, topic, difficulty FROM predictions WHERE task_id = ?', (task_id,))
        result = cursor.fetchone()

        if not result:
            raise ValueError(f"Task ID {task_id} not found")

        user_id, topic, difficulty = result

        # Update predictions table
        cursor.execute('''
        UPDATE predictions
        SET actual_correct = ?, actual_time = ?, updated_at = ?
        WHERE task_id = ?
        ''', (actual_correct, actual_time, datetime.now().isoformat(), task_id))

        # Add to training data
        cursor.execute('''
        INSERT INTO training_data (user_id, topic, difficulty, correct, response_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, topic, difficulty, actual_correct, actual_time, datetime.now().isoformat()))

        self.conn.commit()

    def get_prediction(self, task_id: int) -> Optional[Dict]:
        """Get prediction by task ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM predictions WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()

        if not row:
            return None

        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def get_training_data(self, topic: str) -> pd.DataFrame:
        """Get all training data for a topic"""
        query = '''
        SELECT user_id, topic, difficulty, correct, response_time, timestamp
        FROM training_data
        WHERE topic = ?
        ORDER BY timestamp
        '''
        return pd.read_sql_query(query, self.conn, params=(topic,))

    def get_user_history(self, user_id: str, topic: Optional[str] = None) -> pd.DataFrame:
        """Get user's prediction history"""
        if topic:
            query = '''
            SELECT task_id, topic, difficulty, predicted_correct, predicted_time,
                   actual_correct, actual_time, created_at, updated_at
            FROM predictions
            WHERE user_id = ? AND topic = ?
            ORDER BY created_at DESC
            '''
            return pd.read_sql_query(query, self.conn, params=(user_id, topic))
        else:
            query = '''
            SELECT task_id, topic, difficulty, predicted_correct, predicted_time,
                   actual_correct, actual_time, created_at, updated_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            '''
            return pd.read_sql_query(query, self.conn, params=(user_id,))

    def get_topic_stats(self, topic: str) -> Dict:
        """Get statistics for a topic"""
        cursor = self.conn.cursor()

        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE topic = ?', (topic,))
        total_predictions = cursor.fetchone()[0]

        # Completed predictions
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE topic = ? AND actual_correct IS NOT NULL', (topic,))
        completed = cursor.fetchone()[0]

        # Accuracy by difficulty
        accuracy_by_diff = {}
        for diff in [1, 2, 3]:
            cursor.execute('''
            SELECT AVG(actual_correct)
            FROM predictions
            WHERE topic = ? AND difficulty = ? AND actual_correct IS NOT NULL
            ''', (topic, diff))
            result = cursor.fetchone()[0]
            accuracy_by_diff[diff] = result if result is not None else None

        # Average time by difficulty
        time_by_diff = {}
        for diff in [1, 2, 3]:
            cursor.execute('''
            SELECT AVG(actual_time)
            FROM predictions
            WHERE topic = ? AND difficulty = ? AND actual_time IS NOT NULL
            ''', (topic, diff))
            result = cursor.fetchone()[0]
            time_by_diff[diff] = result if result is not None else None

        return {
            'topic': topic,
            'total_predictions': total_predictions,
            'completed': completed,
            'accuracy_by_difficulty': accuracy_by_diff,
            'avg_time_by_difficulty': time_by_diff
        }

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
