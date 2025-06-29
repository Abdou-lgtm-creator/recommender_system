"""
recommend.py

This script loads a trained Neural Collaborative Filtering (NCF) model
with LLM embeddings and provides personalized book recommendations.
It maps user and book IDs, filters out already-rated books, and
outputs the top 10 recommended books for a given user.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# Load LLM-enhanced model
model = tf.keras.models.load_model("../models/ncf_model_llm.h5")

# Load mappings and metadata
ratings = pd.read_csv("../data/flt_rating.csv")
books = pd.read_csv("../data/books.csv")
user_map = pd.read_csv("../data/user_mapping.csv")
book_map = pd.read_csv("../data/book_mapping.csv")

# Create mapping dictionaries
user2idx = dict(zip(user_map["user_original"], user_map["user_index"]))
book2idx = dict(zip(book_map["book_original"], book_map["book_index"]))
idx2book = dict(zip(book_map["book_index"], book_map["book_original"]))

# Build book metadata table
books["title"] = books["title"].fillna("Unknown")
books["authors"] = books["authors"].fillna("Unknown")
bk_meta = books.set_index("book_id").loc[book2idx.keys(), ["title", "authors"]]
bk_meta.index = [book2idx[bid] for bid in bk_meta.index]

n_books = len(book2idx)


def recommend_for_user(user_id):
    """Recommend and display top 10 books for a given user ID."""
    if user_id not in user2idx:
        print("User ID not found in filtered dataset.")
        return

    user_idx = user2idx[user_id]
    book_indices = np.arange(n_books)
    user_array = np.full(n_books, user_idx)

    scores = model.predict([user_array, book_indices], verbose=0).flatten()

    rated_books = (
        ratings[ratings["user_id"] == user_id]["book_id"]
        .map(book2idx)
        .dropna()
        .astype(int)
        .values
    )
    rf = [i for i in np.argsort(scores)[::-1] if i not in rated_books]
    recommended = rf[:10]

    print(f"\nTop 10 Book Recommendations for User ID {user_id}:")
    for idx in recommended:
        if idx in bk_meta.index:
            row = bk_meta.loc[idx]
            print(f"✔ {row['title']} by {row['authors']}")


if __name__ == "__main__":
    try:
        user_id_input = int(input("Enter a User ID to get recommendations: "))
        recommend_for_user(user_id_input)
    except ValueError:
        print("Please enter a valid numeric User ID.")
