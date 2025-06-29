"""
llm_embeddings.py

Generates and saves LLM embeddings for book items for use in the NCF pipeline.
Falls back to local generation if OpenAI quota or connectivity fails.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

EMBEDDING_MODE = "openai"
ITEMS_CSV_PATH = "../data/books.csv"
OUTPUT_NPY_PATH = "../data/item_embeddings.npy"

try:
    if EMBEDDING_MODE == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            api_key = input("Please enter your OpenAI API key: ").strip()
            if not api_key:
                raise EnvironmentError("OpenAI API key was not provided.")
        client = OpenAI(api_key=api_key)

        def get_embedding(text_input):
            """Generate embedding using OpenAI."""
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-large", input=text_input
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error generating embedding: {e}. Local generation.")
                raise e

except Exception:
    print("Falling back to local embedding generation.")
    EMBEDDING_MODE = "local"

if EMBEDDING_MODE == "local":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df = pd.read_csv(ITEMS_CSV_PATH)
df["title"] = df["title"].fillna("")
df["authors"] = df["authors"].fillna("")
embeddings = []

if EMBEDDING_MODE == "openai":
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Generating OpenAI embeddings"
    ):
        COMBINED_TEXT = f"{row['title']} {row['authors']}"
        try:
            emb = get_embedding(COMBINED_TEXT)
        except Exception:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            )  # noqa: E501
            EMBEDDING_MODE = "local"
            break
        embeddings.append(emb)

if EMBEDDING_MODE == "local":
    texts_to_embed = [
        f"{row['title']} {row['authors']}" for _, row in df.iterrows()
    ]  # noqa: E501
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

embeddings = np.array(embeddings)
np.save(OUTPUT_NPY_PATH, embeddings)
print(f"Saved embeddings to {OUTPUT_NPY_PATH} with shape {embeddings.shape}")
