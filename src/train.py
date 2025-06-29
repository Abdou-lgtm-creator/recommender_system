"""
train.py

This script trains a Neural Collaborative Filtering (NCF) model
using Keras to predict user ratings for books,
with optional LLM embeddings integration.
It ensures the LLM embedding matrix aligns precisely with
the filtered book indices.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load preprocessed data
ratings = pd.read_csv("../data/flt_rating.csv")
user_map = pd.read_csv("../data/user_mapping.csv")
book_map = pd.read_csv("../data/book_mapping.csv")
llm_embeddings_full = np.load("../data/item_embeddings_aligned.npy")

# Encode user and book IDs using mappings
user2idx = dict(zip(user_map["user_original"], user_map["user_index"]))
book2idx = dict(zip(book_map["book_original"], book_map["book_index"]))
ratings["user_idx"] = ratings["user_id"].map(user2idx)
ratings["book_idx"] = ratings["book_id"].map(book2idx)

# Create training data arrays
user_input = ratings["user_idx"].values
book_input = ratings["book_idx"].values
rating_input = ratings["rating"].values

X_train_u, X_test_u, X_train_b, X_test_b, y_train, y_test = train_test_split(
    user_input,
    book_input,
    rating_input,
    test_size=0.2,
    random_state=42,
)

n_users = len(user2idx)
n_books = len(book2idx)

# Align LLM embeddings to filtered book indices
aligned_llm_embeddings = np.zeros(
    (n_books, llm_embeddings_full.shape[1]), dtype=np.float32
)
for book_original, idx in book2idx.items():
    aligned_llm_embeddings[idx] = llm_embeddings_full[int(book_original)]

# Build NCF model with LLM embeddings integration
user_in = Input(shape=(1,), name="user_input")
book_in = Input(shape=(1,), name="book_input")
user_emb = Embedding(n_users, 50)(user_in)

book_emb_layer = Embedding(
    input_dim=n_books,
    output_dim=aligned_llm_embeddings.shape[1],
    weights=[aligned_llm_embeddings],
    trainable=False,
)
book_emb = book_emb_layer(book_in)

x = Concatenate()([Flatten()(user_emb), Flatten()(book_emb)])
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
out = Dense(1)(x)

model = Model(inputs=[user_in, book_in], outputs=out)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# pylint: disable=invalid-name
history = model.fit(
    # pylint: enable=invalid-name
    [X_train_u, X_train_b],
    y_train,
    validation_data=([X_test_u, X_test_b], y_test),
    epochs=10,
    batch_size=512,
    verbose=1,
)

model.save("../models/ncf_model_llm.h5")

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Val Loss", marker="o")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/loss_curve_llm.png")

preds = model.predict([X_test_u, X_test_b], verbose=0).flatten()
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
print(f"\nTest RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")

pd.DataFrame({"Metric": ["RMSE", "MAE"], "Value": [rmse, mae]}).plot(
    kind="bar",
    x="Metric",
    y="Value",
    legend=False,
    title="Test Rating Prediction Error",
)
plt.tight_layout()
plt.savefig("./results/rating_metrics_llm.png")
print(f"Train on {n_users} users × {n_books} books with aligned embeddings")
