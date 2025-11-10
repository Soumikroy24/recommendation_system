from surprise import SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

class SVDppRecommender:
    def __init__(self):
        self.algo = SVDpp()
        self.trainset = None
        self.testset = None

    def train(self, df):
        """Train SVD++ on explicit rating dataset"""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2)
        print("Training SVD++ model...")
        self.algo.fit(self.trainset)
        print(" SVD++ training complete!")

    def evaluate(self):
        """Evaluate with RMSE, MAE, Precision, Recall, Accuracy, and F1-score"""
        predictions = self.algo.test(self.testset)

        # RMSE and MAE
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)

        # Convert to binary for precision/recall
        y_true = np.array([1 if r_ui >= 4 else 0 for (_, _, r_ui, _, _) in predictions])
        y_pred = np.array([1 if est >= 4 else 0 for (_, _, _, est, _) in predictions])

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': acc,
            'F1-Score': f1
        }
        print(" SVD++ Evaluation Metrics:", metrics)
        return metrics

    def predict_rating(self, user_id, book_id):
        """Predict rating for a given user and book"""
        return self.algo.predict(user_id, book_id).est

    def recommend_top_n(self, df, user_id, n=5):
        """Recommend top-n books for a given user based on predicted ratings"""
        all_books = df['book_id'].unique()
        predictions = [(bid, self.predict_rating(user_id, bid)) for bid in all_books]
        top_n = sorted(predictions, key=lambda x: -x[1])[:n]
        return [int(bid) for bid, _ in top_n]
