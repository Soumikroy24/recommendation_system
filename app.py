from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from models.svdpp_model.svdpp_recommender import SVDppRecommender
from models.bpr_model.bpr_recommender import BPRRecommender
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Global models and data
svdpp_model = None
bpr_model = None
dataset = None
image_cache = {}

# --------------------------------------------------------
#  HOME ROUTE — SERVE FRONTEND
# --------------------------------------------------------

@app.route('/')
def index():
    """Render the interactive book recommendation UI"""
    return render_template('index.html')


# --------------------------------------------------------
#  TRAIN BOTH MODELS (SVD++ & BPR)
# --------------------------------------------------------

@app.route('/train', methods=['POST'])
def train_models():
    global svdpp_model, bpr_model, dataset

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Please upload a dataset (ratings.csv)'}), 400

    dataset = pd.read_csv(file)
    required_cols = {'user_id', 'book_id', 'rating'}
    if not required_cols.issubset(dataset.columns):
        return jsonify({'error': 'Dataset must contain user_id, book_id, and rating columns'}), 400

    # --- Train SVD++ ---
    svdpp_model = SVDppRecommender()
    svdpp_model.train(dataset)
    svd_metrics = svdpp_model.evaluate()

    # --- Train BPR ---
    bpr_model = BPRRecommender()
    implicit_df = bpr_model.convert_to_implicit(dataset)
    bpr_model.build_matrix(implicit_df)
    bpr_model.train()
    bpr_metrics = bpr_model.evaluate(k=10)

    return jsonify({
        'message': ' Both models trained successfully!',
        'SVD++_Metrics': svd_metrics,
        'BPR_Metrics': bpr_metrics
    })


# --------------------------------------------------------
#  IMAGE ENHANCEMENT FUNCTION
# --------------------------------------------------------

def get_cover_image(book):
    """
    Smart image fetcher:
    1 Skip Goodreads 'gray G' placeholders
    2 Try Google Books API (high-res)
    3 Try OpenLibrary cover
    4 Fallback to local static 'no_cover.png'
    """
    # Step 1: Check Goodreads images
    img = book.get("image_url") or book.get("small_image_url")
    bad_patterns = [
        "books/1320399351m/",
        "nophoto",
        "no_cover",
        "goodreads.com/assets"
    ]
    if isinstance(img, str) and any(p in img for p in bad_patterns):
        img = None

    # Step 2: If valid image
    if isinstance(img, str) and img.startswith("http"):
        return img

    # Step 3: Lookup ISBN
    isbn = str(book.get("isbn", "")).strip()
    if isbn in image_cache:
        return image_cache[isbn]

    # Step 4: Try Google Books API
    if isbn and isbn.lower() != "nan":
        try:
            res = requests.get(f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}", timeout=5)
            data = res.json()
            if "items" in data and "imageLinks" in data["items"][0]["volumeInfo"]:
                links = data["items"][0]["volumeInfo"]["imageLinks"]
                img_url = links.get("extraLarge") or links.get("large") or links.get("medium") or links.get("thumbnail")
                if img_url:
                    img_url = img_url.replace("&zoom=1", "&zoom=3")
                    image_cache[isbn] = img_url
                    return img_url
        except Exception:
            pass

    # Step 5: Try OpenLibrary
    if isbn:
        img_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
        image_cache[isbn] = img_url
        return img_url

    # Step 6: Fallback to local no_cover image
    return "/static/assets/no_cover.png"


# --------------------------------------------------------
#  INDIVIDUAL MODEL RECOMMENDATIONS
# --------------------------------------------------------

@app.route('/recommend/svdpp')
def recommend_svdpp():
    global svdpp_model, dataset
    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=5, type=int)
    if svdpp_model is None:
        return jsonify({'error': 'SVD++ model not trained yet'}), 500

    recs = svdpp_model.recommend_top_n(dataset, user_id, n)
    return jsonify({'user_id': user_id, 'recommendations': recs})


@app.route('/recommend/bpr')
def recommend_bpr():
    global bpr_model
    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=5, type=int)
    if bpr_model is None:
        return jsonify({'error': 'BPR model not trained yet'}), 500

    recs = bpr_model.recommend(user_id, n)
    return jsonify({'user_id': user_id, 'recommendations': recs})


# --------------------------------------------------------
#  COMBINED USER RECOMMENDATION ROUTE
# --------------------------------------------------------

@app.route('/recommend/user', methods=['GET'])
def recommend_for_user():
    """
    Generate top-N book recommendations for a given user using both SVD++ and BPR models.
    Returns consistent JSON for the frontend (id, title, author, rating, image_url).
    """
    global svdpp_model, bpr_model, dataset

    user_id = request.args.get('user_id', type=int)
    n = request.args.get('n', default=8, type=int)

    if svdpp_model is None or bpr_model is None:
        return jsonify({'error': ' Models not trained yet. Please POST to /train first.'}), 500

    try:
        #  Load dataset
        books_df = pd.read_csv("data/books.csv")

        #  Build lookup dictionary only using book_id
        book_map = {}
        for _, row in books_df.iterrows():
            bid = row.get("book_id")
            if pd.notna(bid):
                bid = int(bid)
                book_map[bid] = {
                    "id": bid,
                    "title": row.get("title", f"Book ID {bid}"),
                    "author": row.get("authors", "Unknown"),
                    "rating": float(row.get("average_rating", 0) or 0),
                    "image_url": get_cover_image(row)
                }

        #  Get SVD++ recommendations
        svd_recs = svdpp_model.recommend_top_n(dataset, user_id, n)
        svd_data = []
        for bid in svd_recs:
            if int(bid) in book_map:
                svd_data.append(book_map[int(bid)])
            else:
                svd_data.append({
                    "id": int(bid),
                    "title": f"Book ID {bid}",
                    "author": "Unknown",
                    "rating": 0,
                    "image_url": "/static/assets/no_cover.png"
                })

        #  Get BPR recommendations
        bpr_recs = bpr_model.recommend(user_id, n)
        bpr_data = []
        for bid in bpr_recs:
            if int(bid) in book_map:
                bpr_data.append(book_map[int(bid)])
            else:
                bpr_data.append({
                    "id": int(bid),
                    "title": f"Book ID {bid}",
                    "author": "Unknown",
                    "rating": 0,
                    "image_url": "/static/assets/no_cover.png"
                })

        #  Return consistent JSON response
        return jsonify({
            "user_id": user_id,
            "SVD++_Recommendations": svd_data,
            "BPR_Recommendations": bpr_data
        })

    except Exception as e:
        print(f" ERROR in /recommend/user: {e}")  # Logs visible in terminal
        return jsonify({'error': str(e)}), 500



# --------------------------------------------------------
#  SIMILAR BOOKS RECOMMENDATION ROUTE
# --------------------------------------------------------

@app.route('/recommend/similar', methods=['GET'])
def recommend_similar_books():
    """Recommend similar books based on shared tags using cosine similarity."""
    try:
        book_id = request.args.get('book_id', type=int)
        top_n = request.args.get('n', default=5, type=int)

        # Load data
        books = pd.read_csv("data/books.csv")
        book_tags = pd.read_csv("data/book_tags.csv")
        tags = pd.read_csv("data/tags.csv")

        # Merge and pivot
        book_tags = book_tags.merge(tags, on='tag_id', how='left')
        pivot = book_tags.pivot_table(index='goodreads_book_id', columns='tag_name', values='count', fill_value=0)
        sim_matrix = cosine_similarity(pivot)

        if book_id not in books['book_id'].values:
            return jsonify({'error': f'Book ID {book_id} not found in books.csv'}), 404

        # Get corresponding goodreads ID
        book_row = books[books['book_id'] == book_id]
        goodreads_id = int(book_row['book_id'].values[0])

        if goodreads_id not in pivot.index:
            return jsonify({'error': f'Book ID {book_id} has no tag data'}), 404

        # Compute similarities
        idx = list(pivot.index).index(goodreads_id)
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        similar_ids = [pivot.index[i] for i, _ in sim_scores]
        similar_books = books[books['book_id'].isin(similar_ids)][['book_id', 'title']].head(top_n)
        similar_books_list = similar_books['title'].tolist()

        return jsonify({'book_id': book_id, 'Similar_Books': similar_books_list})

    except Exception as e:
        return jsonify({'error': str(e)})


# --------------------------------------------------------
#  APP ENTRY POINT
# --------------------------------------------------------

if __name__ == '__main__':
    print(" Starting Flask server on http://127.0.0.1:5000 ...")
    # Disable auto reload + disable debugger to prevent duplicate training
    app.run(debug=False, use_reloader=False)

