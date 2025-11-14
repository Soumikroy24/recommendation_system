import os
import joblib
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
        df['interaction'] = (df['rating'] >= threshold).astype(int)
        df = df[df['interaction'] == 1]
        self.df = df
        print(f"Converted to implicit feedback: {len(df)} positive interactions")
        return df

    def build_matrix(self, df):
        # SORT ensures stable mapping everywhere
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['book_id'].unique())

        # user_id → row index
        self.user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.rev_user_map = {idx: user_id for user_id, idx in self.user_map.items()}

        # book_id → col index
        self.item_map = {book_id: idx for idx, book_id in enumerate(unique_items)}
        self.rev_item_map = {idx: book_id for book_id, idx in self.item_map.items()}

        rows = df['user_id'].map(self.user_map)
        cols = df['book_id'].map(self.item_map)
        data = np.ones(len(df))

        self.interaction_matrix = coo_matrix((data, (rows, cols))).tocsr()
        print(f"Matrix built: {self.interaction_matrix.shape} (CSR format)")

    def train(self):
        print("Training BPR model...")
        self.model.fit(self.interaction_matrix)
        print("BPR training complete!")

    def recommend(self, user_id, n=5):
        if user_id not in self.user_map:
            return []

        uidx = self.user_map[user_id]
        recs, _ = self.model.recommend(uidx, self.interaction_matrix[uidx], N=n)

        # return REAL book_ids, not internal ones
        return [int(self.rev_item_map[i]) for i in recs if i in self.rev_item_map]

    def evaluate(self, k=10):
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

        return {
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'Accuracy': np.mean(accuracies),
            'F1-Score': np.mean(f1s)
        }

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'rev_user_map': self.rev_user_map,
            'rev_item_map': self.rev_item_map
        }, path)
        print(f"BPR model saved to {path}")

    def load_model(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.user_map = data['user_map']
        self.item_map = data['item_map']
        self.rev_user_map = data['rev_user_map']
        self.rev_item_map = data['rev_item_map']
        print(f"BPR model loaded from {path}")
