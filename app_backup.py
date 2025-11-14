from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from models.svdpp_model.svdpp_recommender import SVDppRecommender
from models.bpr_model.bpr_recommender import BPRRecommender
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.config.update({
    "JSONIFY_PRETTYPRINT_REGULAR": True,
    "JSON_SORT_KEYS": False
})

# ============================================================== 
#  AUTO-LOAD MODELS AND DATA AT STARTUP
# ==============================================================

svdpp_model, bpr_model, dataset = None, None, None
image_cache = {}

try:
    if os.path.exists("models/svdpp_model.pkl"):
        svdpp_model = SVDppRecommender()
        svdpp_model.load_model("models/svdpp_model.pkl")
        print("Loaded SVD++ model from disk")

    if os.path.exists("models/bpr_model.pkl"):
        bpr_model = BPRRecommender()
        bpr_model.load_model("models/bpr_model.pkl")
        print("Loaded BPR model from disk")

except Exception as e:
    print(f"Could not load models: {e}")

# 🔹 Load dataset (ratings.csv)
if os.path.exists("data/ratings.csv"):
    try:
        dataset = pd.read_csv("data/ratings.csv")
        print(f"Loaded dataset with {len(dataset)} rows")
    except Exception as e:
        print(f"Failed to load dataset: {e}")

# 🔹 Rebuild BPR interaction matrix if model present
if bpr_model and dataset is not None:
    try:
        df_implicit = bpr_model.convert_to_implicit(dataset.copy())
        bpr_model.build_matrix(df_implicit)
        print("Rebuilt BPR interaction matrix from ratings.csv")
    except Exception as e:
        print(f"Failed to rebuild BPR matrix: {e}")


# ============================================================== 
#  ROUTES
# ==============================================================

@app.route('/')
def index():
    """Home route — render main UI"""
    return render_template('index.html')


# ============================================================== 
#  TRAIN MODELS
# ==============================================================

@app.route('/train', methods=['POST'])
def train_models():
    global svdpp_model, bpr_model, dataset

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Please upload ratings.csv'}), 400

    dataset = pd.read_csv(file)
    required_cols = {'user_id', 'book_id', 'rating'}
    if not required_cols.issubset(dataset.columns):
        return jsonify({'error': 'Dataset must contain user_id, book_id, rating'}), 400

    print("Training started...")

    # --- Train SVD++ ---
    svdpp_model = SVDppRecommender()
    svdpp_model.train(dataset)
    svd_metrics = svdpp_model.evaluate()

    # --- Train BPR ---
    bpr_model = BPRRecommender()
    implicit_df = bpr_model.convert_to_implicit(dataset.copy())
    bpr_model.build_matrix(implicit_df)
    bpr_model.train()
    bpr_metrics = bpr_model.evaluate(k=10)

    os.makedirs("models", exist_ok=True)
    svdpp_model.save_model("models/svdpp_model.pkl")
    bpr_model.save_model("models/bpr_model.pkl")

    print("Training complete and models saved.")
    return jsonify({
        'message': 'Both models trained and saved successfully!',
        'SVD++_Metrics': svd_metrics,
        'BPR_Metrics': bpr_metrics
    })


# ============================================================== 
#  IMAGE FETCH HELPER
# ==============================================================

def get_cover_image(book_row):
    """Return best available cover image or local fallback.
       'book_row' is a pandas Series / dict-like representing a books.csv row."""
    img = None
    # try common names used in our books.csv
    for key in ("image_url", "small_image_url", "image"):
        if isinstance(book_row.get(key, None), str) and book_row.get(key).startswith("http"):
            img = book_row.get(key)
            break

    bad_patterns = ["books/1320399351m/", "nophoto", "no_cover", "goodreads.com/assets"]
    if isinstance(img, str) and any(p in img for p in bad_patterns):
        img = None

    if isinstance(img, str) and img.startswith("http"):
        return img

    # fallback to OpenLibrary by ISBN if available
    isbn = str(book_row.get("isbn", "")).strip()
    if isbn and isbn not in ("nan", ""):
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"

    return "/static/assets/no_cover.png"


# ============================================================== 
#  INDIVIDUAL MODEL ROUTES
# ==============================================================

@app.route('/recommend/svdpp')
def recommend_svdpp():
    global svdpp_model, dataset
    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=5, type=int)

    if svdpp_model is None:
        return jsonify({'error': '❌ SVD++ model not loaded'}), 500
    if dataset is None:
        return jsonify({'error': '❌ Dataset not loaded'}), 500

    try:
        recs = svdpp_model.recommend_top_n(dataset, user_id, n)
        return jsonify({'user_id': user_id, 'recommendations': recs})
    except Exception as e:
        print(f"ERROR in /recommend/svdpp: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommend/bpr')
def recommend_bpr():
    global bpr_model
    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=5, type=int)

    if bpr_model is None:
        return jsonify({'error': '❌ BPR model not loaded'}), 500

    try:
        recs = bpr_model.recommend(user_id, n)
        return jsonify({'user_id': user_id, 'recommendations': recs})
    except Exception as e:
        print(f"ERROR in /recommend/bpr: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================== 
#  COMBINED RECOMMENDER
# ==============================================================

@app.route('/recommend/user', methods=['GET'])
def recommend_for_user():
    global svdpp_model, bpr_model, dataset

    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=8, type=int)

    if svdpp_model is None or bpr_model is None:
        return jsonify({'error': 'Models not trained yet.'}), 500

    print(f"Generating recommendations for user {user_id}...")

    try:
        # Use 'id' column from books.csv as the canonical book id, and rename to 'book_id'
        books_df = pd.read_csv("data/books.csv")[
            ['id', 'title', 'authors', 'average_rating', 'image_url', 'small_image_url', 'isbn']
        ].rename(columns={'id': 'book_id'})

        # Build a lookup map: ratings.book_id (which corresponds to books.csv.id) -> metadata
        book_map = {}
        for _, row in books_df.iterrows():
            bid = int(row['book_id'])
            book_map[bid] = {
                "id": bid,
                "title": str(row.get("title", "") or ""),
                "author": str(row.get("authors", "Unknown") or "Unknown"),
                "rating": float(row.get("average_rating", 0) or 0),
                "image_url": get_cover_image(row)
            }

        svd_ids = svdpp_model.recommend_top_n(dataset, user_id, n)
        bpr_ids = bpr_model.recommend(user_id, n)

        svd_out = [book_map.get(bid, {
            "id": bid, "title": f"Book ID {bid}", "author": "Unknown",
            "rating": 0, "image_url": "/static/assets/no_cover.png"
        }) for bid in svd_ids]

        bpr_out = [book_map.get(bid, {
            "id": bid, "title": f"Book ID {bid}", "author": "Unknown",
            "rating": 0, "image_url": "/static/assets/no_cover.png"
        }) for bid in bpr_ids]

        return jsonify({
            "user_id": user_id,
            "message": f"Recommendations generated for user {user_id}",
            "SVD++_Recommendations": svd_out,
            "BPR_Recommendations": bpr_out
        })

    except Exception as e:
        print(f"ERROR in /recommend/user: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================== 
#  SIMILAR BOOKS
# ==============================================================

@app.route('/recommend/similar')
def recommend_similar():
    """Return structured similar books (title + author)."""
    try:
        book_id = request.args.get('book_id', type=int)
        n = request.args.get('n', default=5, type=int)

        # Load books with both id and best_book_id so we can map goodreads ids -> our canonical id
        books = pd.read_csv("data/books.csv")[['id', 'best_book_id', 'title', 'authors']].rename(columns={'id': 'book_id'})
        book_tags = pd.read_csv("data/book_tags.csv")
        tags = pd.read_csv("data/tags.csv")

        # Merge tag names
        book_tags = book_tags.merge(tags, on='tag_id', how='left')

        # Pivot uses the Goodreads-style id column (goodreads_book_id) from book_tags
        pivot = book_tags.pivot_table(index='goodreads_book_id', columns='tag_name', values='count', fill_value=0)

        # Map pivot index (goodreads_book_id / best_book_id) -> our canonical book_id
        # build mapping: best_book_id -> book_id
        mapping = dict(zip(books['best_book_id'].astype('Int64').fillna(-1).astype(int), books['book_id'].astype(int)))

        # Map pivot index to our ids where possible
        pivot = pivot.rename(index=mapping)

        # Drop any rows that couldn't be mapped (index not in mapping)
        pivot = pivot[~pivot.index.isnull()]
        pivot.index = pivot.index.astype(int)
        pivot = pivot[~pivot.index.duplicated(keep='first')]

        if book_id not in pivot.index:
            # If requested book_id is not present in pivot (no tag info), return a helpful error
            return jsonify({'error': f'No tag information available for book {book_id}'}), 404

        # Compute similarity on the pivot (rows are canonical book_ids)
        sim_matrix = cosine_similarity(pivot)

        idx = list(pivot.index).index(book_id)
        sim_scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
        similar_ids = [pivot.index[i] for i, _ in sim_scores]

        # Build response using books dataframe
        resp_books = books[books['book_id'].isin(similar_ids)][['book_id', 'title', 'authors']]
        resp_books = resp_books.drop_duplicates(subset=['book_id'])

        return jsonify({
            "book_id": int(book_id),
            "title": books.loc[books['book_id'] == book_id, 'title'].iloc[0] if book_id in list(books['book_id']) else "",
            "Similar_Books": resp_books.to_dict(orient='records')
        })

    except Exception as e:
        print(f"ERROR in /recommend/similar: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================== 
# 🩺 HEALTH CHECK
# ==============================================================

@app.route('/health')
def health():
    return jsonify({"status": "ok"})


# ============================================================== 
#  RUN APP
# ==============================================================

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(debug=False, use_reloader=False)
