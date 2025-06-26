"""
preprocessing.py

This module prepares the dataset for training a Neural Collaborative Filtering (NCF) model.
It performs the following key steps:
- Loads raw ratings and book metadata
- Filters users and books based on activity thresholds (25th percentile)
- Encodes user_id and book_id into numeric indices
- Constructs a sparse user-item interaction matrix
- Saves mappings and processed files for use in model training and recommendation

This preprocessing is optimized for implicit and explicit feedback systems using deep learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz
import os


def load_dataset(ratings_path: str, books_path: str) -> tuple:
    """
    Load ratings and books dataset from CSV files.

    Args:
        ratings_path (str): Path to the ratings CSV file.
        books_path (str): Path to the books CSV file.

    Returns:
        tuple: DataFrames of ratings and books.
    """
    ratings = pd.read_csv('../data/ratings.csv')
    books = pd.read_csv('../data/books.csv')
    return ratings, books


def filter_ratings_by_quantile(ratings: pd.DataFrame, user_quantile: float = 0.25, book_quantile: float = 0.25) -> pd.DataFrame:
    """
    Filter out users and books below the specified quantile thresholds based on number of ratings.

    Args:
        ratings (DataFrame): Ratings data.
        user_quantile (float): Quantile threshold for user filtering.
        book_quantile (float): Quantile threshold for book filtering.

    Returns:
        DataFrame: Filtered ratings data.
    """
    # Filter users
    user_counts = ratings['user_id'].value_counts()
    user_threshold = user_counts.quantile(user_quantile)
    active_users = user_counts[user_counts >= user_threshold].index
    ratings = ratings[ratings['user_id'].isin(active_users)]

    # Filter books
    book_counts = ratings['book_id'].value_counts()
    book_threshold = book_counts.quantile(book_quantile)
    popular_books = book_counts[book_counts >= book_threshold].index
    ratings = ratings[ratings['book_id'].isin(popular_books)]

    # Print filtering summary
    print(f"[INFO] Original users: {user_counts.shape[0]}")
    print(f"[INFO] Filtered users: {ratings['user_id'].nunique()}")
    print(
        f"[INFO] User threshold (quantile {user_quantile}): {user_threshold}")

    print(f"[INFO] Original books: {book_counts.shape[0]}")
    print(f"[INFO] Filtered books: {ratings['book_id'].nunique()}")
    print(
        f"[INFO] Book threshold (quantile {book_quantile}): {book_threshold}")

    return ratings


def encode_ids(ratings: pd.DataFrame) -> tuple:
    """
    Encode user_id and book_id into consecutive integer indices.

    Args:
        ratings (DataFrame): Ratings data with original IDs.

    Returns:
        tuple: Updated ratings, user LabelEncoder, book LabelEncoder
    """
    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()

    ratings['user_index'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['book_index'] = book_encoder.fit_transform(ratings['book_id'])

    return ratings, user_encoder, book_encoder


def build_user_item_matrix(ratings: pd.DataFrame) -> csr_matrix:
    """
    Construct the sparse user-item interaction matrix in CSR format.

    Args:
        ratings (DataFrame): Ratings data with encoded indices.

    Returns:
        csr_matrix: User-item sparse matrix.
    """
    return csr_matrix((ratings['rating'], (ratings['user_index'], ratings['book_index'])))


def save_sparse_matrix(matrix: csr_matrix, path: str):
    """
    Save sparse matrix to .npz format.

    Args:
        matrix (csr_matrix): Sparse matrix to save.
        path (str): Destination file path.
    """
    save_npz(path, matrix)
    print(f"[SAVED] Sparse matrix → {path}")


def save_mapping(enc: LabelEncoder, name: str, path: str):
    """
    Save ID mapping (original → encoded) as a CSV.

    Args:
        enc (LabelEncoder): Trained encoder.
        name (str): 'user' or 'book'
        path (str): Output file path.
    """
    mapping = pd.DataFrame({
        f'{name}_original': enc.classes_,
        f'{name}_index': range(len(enc.classes_))
    })
    mapping.to_csv(path, index=False)
    print(f"[SAVED] {name.capitalize()} mapping → {path}")


# === Main Execution ===
if __name__ == "__main__":
    # Create output directory
    os.makedirs("../data", exist_ok=True)

    # Step 1: Load datasets
    ratings, books = load_dataset("ratings.csv", "books.csv")

    # Step 2: Filter by rating activity thresholds
    ratings = filter_ratings_by_quantile(
        ratings, user_quantile=0.25, book_quantile=0.25)

    # Step 3: Encode user_id and book_id
    ratings, user_enc, book_enc = encode_ids(ratings)

    # Step 4: Build user-item sparse matrix
    matrix = build_user_item_matrix(ratings)

    # Step 5: Save artifacts for modeling
    save_sparse_matrix(matrix, "../data/ratings_matrix.npz")
    save_mapping(user_enc, "user", "../data/user_mapping.csv")
    save_mapping(book_enc, "book", "../data/book_mapping.csv")
    ratings.to_csv("../data/filtered_ratings.csv", index=False)
    print(f"[SAVED] Filtered and encoded ratings → data/filtered_ratings.csv")

    # Shape of the matrix
    print(f"[INFO] Matrix shape: {matrix.shape} (users × books)")
    print("[DONE] Preprocessing complete!")
