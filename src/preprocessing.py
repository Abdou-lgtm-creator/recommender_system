"""
preprocessing.py

Prepares the dataset for training a Neural Collaborative Filtering (NCF)
model with LLM embeddings.
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder


def load_dataset(ratings_path: str, books_path: str) -> tuple:
    """Load ratings and books dataset from CSV files."""
    ratings = pd.read_csv(ratings_path)
    books = pd.read_csv(books_path)
    return ratings, books


def filter_ratings_by_quantile(
    ratings_data: pd.DataFrame,
    user_quantile: float = 0.25,
    book_quantile: float = 0.25,
) -> pd.DataFrame:
    """Filter out users and books below quantile thresholds."""
    user_counts = ratings_data["user_id"].value_counts()
    user_thresh = user_counts.quantile(user_quantile)
    active_users = user_counts[user_counts >= user_thresh].index
    filtered = ratings_data[ratings_data["user_id"].isin(active_users)]

    book_counts = filtered["book_id"].value_counts()
    book_thresh = book_counts.quantile(book_quantile)
    popular_books = book_counts[book_counts >= book_thresh].index
    filtered = filtered[filtered["book_id"].isin(popular_books)]

    print("Original users:", user_counts.shape[0])
    print("Filtered users:", filtered["user_id"].nunique())
    print(f"User threshold ({user_quantile}): {user_thresh:.2f}")
    print("Original books:", book_counts.shape[0])
    print("Filtered books:", filtered["book_id"].nunique())
    print(f"Book threshold ({book_quantile}): {book_thresh:.2f}")

    return filtered


def encode_ids(flt_rating: pd.DataFrame) -> tuple:
    """Encode user_id and book_id into integer indices."""
    u_enc = LabelEncoder()
    b_enc = LabelEncoder()
    flt_rating["user_index"] = u_enc.fit_transform(flt_rating["user_id"])
    flt_rating["book_index"] = b_enc.fit_transform(flt_rating["book_id"])
    return flt_rating, u_enc, b_enc


def build_user_item_matrix(encoded_ratings: pd.DataFrame) -> csr_matrix:
    """Construct user-item interaction matrix in CSR format."""
    return csr_matrix(
        (
            encoded_ratings["rating"],
            (
                encoded_ratings["user_index"],
                encoded_ratings["book_index"],
            ),
        )
    )


def save_sparse_matrix(sparse_matrix: csr_matrix, path: str):
    """Save sparse matrix in .npz format."""
    save_npz(path, sparse_matrix)
    print("[SAVED] Sparse matrix saved to", path)


def save_mapping(encoder: LabelEncoder, name: str, path: str):
    """Save ID mapping as CSV."""
    mapping = pd.DataFrame(
        {
            f"{name}_original": encoder.classes_,
            f"{name}_index": range(len(encoder.classes_)),
        }
    )
    mapping.to_csv(path, index=False)
    print(f"[SAVED] {name.capitalize()} mapping saved to", path)


def align_and_save_llm_embeddings(
    books_path: str,
    embeddings_path: str,
    output_path: str,
):
    """Align and save LLM item embeddings with book order."""
    books = pd.read_csv(books_path)
    embeddings = np.load(embeddings_path)
    assert len(books) == len(
        embeddings
    ), f"Mismatch: {len(books)} books vs {len(embeddings)} embeddings"
    np.save(output_path, embeddings)
    print("[SAVED] Aligned LLM embeddings saved to", output_path)


def run_preprocessing():
    """Run the preprocessing pipeline with LLM embeddings."""
    ratings, _ = load_dataset("../data/ratings.csv", "../data/books.csv")
    filtered = filter_ratings_by_quantile(ratings)
    encoded, u_enc, b_enc = encode_ids(filtered)
    matrix = build_user_item_matrix(encoded)

    save_sparse_matrix(matrix, "../data/ratings_matrix.npz")
    save_mapping(u_enc, "user", "../data/user_mapping.csv")
    save_mapping(b_enc, "book", "../data/book_mapping.csv")
    encoded.to_csv("../data/flt_rating.csv", index=False)

    align_and_save_llm_embeddings(
        "../data/books.csv",
        "../data/item_embeddings.npy",
        "../data/item_embeddings_aligned.npy",
    )

    print("[SAVED] Filtered and encoded ratings saved.")
    print(f"[INFO] Matrix shape: {matrix.shape} (users × books)")
    print("[DONE] Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
