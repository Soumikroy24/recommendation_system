import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

class BPRRecommender:
    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01):
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization
        )
        self.user_map = {}
        self.item_map = {}
        self.rev_user_map = {}
        self.rev_item_map = {}
        self.interaction_matrix = None
        self.df = None

    def convert_to_implicit(self, df, threshold=4):
        """Convert explicit ratings (1–5) into implicit binary interactions"""
        df['interaction'] = (df['rating'] >= threshold).astype(int)
        df = df[df['interaction'] == 1]
        self.df = df
        print(f"Converted to implicit feedback: {len(df)} positive interactions")
        return df

    def build_matrix(self, df):
        """Build sparse user-item interaction matrix"""
        self.user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
        self.item_map = {i: j for j, i in enumerate(df['book_id'].unique())}
        self.rev_user_map = {v: k for k, v in self.user_map.items()}
        self.rev_item_map = {v: k for k, v in self.item_map.items()}

        rows = df['user_id'].map(self.user_map)
        cols = df['book_id'].map(self.item_map)
        data = np.ones(len(df))

        self.interaction_matrix = coo_matrix((data, (rows, cols))).tocsr()
        print(f" Matrix built: {self.interaction_matrix.shape} (CSR format)")

    def train(self):
        """Train BPR model"""
        print("Training BPR model...")
        self.model.fit(self.interaction_matrix)
        print(" BPR training complete!")

    def recommend(self, user_id, n=5):
        """Recommend top-n books for a user"""
        if user_id not in self.user_map:
            return []
        uidx = self.user_map[user_id]
        recs, _ = self.model.recommend(uidx, self.interaction_matrix[uidx], N=n)
        return [int(self.rev_item_map[i]) for i in recs]

    def evaluate(self, k=10):
        """Evaluate BPR with Precision, Recall, Accuracy, and F1-score"""
        precisions, recalls, accuracies, f1s = [], [], [], []
        users = list(self.user_map.keys())

        for u in users:
            true_items = set(self.df[self.df['user_id'] == u]['book_id'])
            recommended = set(self.recommend(u, n=k))
            if not true_items:
                continue

            tp = len(true_items & recommended)
            fp = len(recommended - true_items)
            fn = len(true_items - recommended)
            tn = max(0, (len(self.item_map) - tp - fp - fn))

            y_true = [1]*tp + [0]*fp + [1]*fn + [0]*tn
            y_pred = [1]*tp + [1]*fp + [0]*fn + [0]*tn

            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        metrics = {
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'Accuracy': np.mean(accuracies),
            'F1-Score': np.mean(f1s)
        }
        print(" BPR Evaluation Metrics:", metrics)
        return metrics
