import os
import joblib
import numpy as np
from functools import lru_cache
from surprise import SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

class SVDppRecommender:
    def __init__(self):
        self.algo = SVDpp()
        self.trainset = None
        self.testset = None
        self.all_book_ids = None  # REAL book ids

    def train(self, df):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)

        self.trainset, self.testset = train_test_split(data, test_size=0.2)
        self.all_book_ids = sorted(df['book_id'].unique())

        print("Training SVD++...")
        self.algo.fit(self.trainset)
        print("SVD++ training complete.")

    def evaluate(self):
        predictions = self.algo.test(self.testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        y_true = np.array([1 if r_ui >= 4 else 0 for (_, _, r_ui, _, _) in predictions])
        y_pred = np.array([1 if est >= 4 else 0 for (_, _, _, est, _) in predictions])

        return {
            'RMSE': rmse,
            'MAE': mae,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0)
        }

    @lru_cache(maxsize=200000)
    def predict_rating(self, user_id, book_id):
        return self.algo.predict(user_id, book_id).est

    def recommend_top_n(self, df, user_id, n=5):
        rated_books = set(df[df['user_id'] == user_id]['book_id'])

        predictions = [
            (bid, self.predict_rating(user_id, bid))
            for bid in self.all_book_ids
            if bid not in rated_books   # ❗ Exclude already rated
        ]

        top_n = sorted(predictions, key=lambda x: -x[1])[:n]
        return [int(bid) for bid, _ in top_n]

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'algo': self.algo,
            'all_book_ids': self.all_book_ids
        }, path)
        print(f"SVD++ model saved to {path}")

    def load_model(self, path):
        data = joblib.load(path)
        self.algo = data['algo']
        self.all_book_ids = data['all_book_ids']
        print(f"SVD++ model loaded from {path}")
