"""
app.py

Streamlit app for personalized book recommendations using a trained
Neural Collaborative Filtering (NCF) model.
Users can receive top book suggestions based on past ratings,
with additional filter options and surprise picks.
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# === Language Code Map ===
LANGUAGE_MAP = {
    "eng": "English",
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "spa": "Spanish",
    "fre": "French",
    "ger": "German",
    "ita": "Italian",
    "por": "Portuguese",
    "nl": "Dutch",
    "mul": "Multiple",
    "": "Unknown",
}

# Background image URL
PIC = "https://images.unsplash.com/photo-1507842217343-583bb7270b66?fit=crop"

# === Load model and data ===


@st.cache_resource
def load_model():
    """Load the pre-trained NCF model."""
    return tf.keras.models.load_model("../models/ncf_model_llm.h5")


@st.cache_data
def load_data():
    """Load ratings, books, user and book mappings."""
    ratings_df = pd.read_csv("../data/flt_rating.csv")
    books_df = pd.read_csv("../data/books.csv")
    user_map_df = pd.read_csv("../data/user_mapping.csv")
    book_map_df = pd.read_csv("../data/book_mapping.csv")

    books_df = books_df.loc[
        books_df["small_image_url"].str.startswith("http", na=False)
    ]

    lang_mapped = books_df["language_code"].map(LANGUAGE_MAP)
    books_df["language_name"] = lang_mapped.fillna(books_df["language_code"])

    return ratings_df, books_df, user_map_df, book_map_df


model = load_model()
ratings, books, user_map, book_map = load_data()

# === Mappings ===
user2idx = dict(zip(user_map["user_original"], user_map["user_index"]))
book2idx = dict(zip(book_map["book_original"], book_map["book_index"]))
book_idx2meta = books.set_index("book_id")[
    [
        "title",
        "authors",
        "small_image_url",
        "original_publication_year",
        "average_rating",
        "language_code",
        "language_name",
    ]
]

# === Styling ===
st.set_page_config(page_title="📚 Book Recommender", layout="centered")

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stApp"] {{
        background-image: url('{PIC}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label,
    .stText, .stMarkdown, .stButton > button {{
        color: white !important;
    }}
    .stButton>button {{
        background-color: #ffffff22;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid #fff;
        font-weight: bold;
    }}
    [data-baseweb="select"] {{
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 500;
    }}
    [data-baseweb="menu"] {{
        background-color: rgba(0, 0, 0, 0.85) !important;
    }}
    [data-baseweb="menu"] div[role="option"] {{
        color: white !important;
    }}
    [data-baseweb="menu"] div[role="option"]:hover {{
        background-color: #444 !important;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === App Header ===
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>NextChapter</h1>
        <p style='font-size: 20px;'>Discover books tailored just for you 📖</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# === Select User ===
user_ids = list(user2idx.keys())
selected_user = st.selectbox("Select a User ID", user_ids)

# === Filter Options ===
st.sidebar.header("📂 Filter Options")

lang_options = books["language_name"].dropna().unique()
selected_lang = st.sidebar.selectbox("Language", sorted(lang_options))

MIN_YEAR = 0
max_year = int(books["original_publication_year"].max())
selected_year_range = st.sidebar.slider(
    "Publication Year", MIN_YEAR, max_year, (2000, 2020)
)

min_rating = st.sidebar.slider("Minimum Average Rating", 1.0, 5.0, 3.5)

# Placeholder for missing image URLs
DEFAULT_IMAGE = "https://via.placeholder.com/80x120.png?text=No+Image"

# === Generate Recommendations ===
if selected_user:
    user_idx = user2idx[selected_user]
    n_books = len(book2idx)
    book_indices = np.arange(n_books)
    user_array = np.full(n_books, user_idx)

    scores = model.predict([user_array, book_indices], verbose=0).flatten()
    rated_books = (
        ratings[ratings["user_id"] == selected_user]["book_id"]
        .map(book2idx)
        .dropna()
        .astype(int)
        .values
    )

    recommended = [i for i in np.argsort(scores)[::-1] if i not in rated_books]
    year_condition = book_idx2meta["original_publication_year"].between(
        *selected_year_range
    )

    filtered_books = book_idx2meta[
        (book_idx2meta["language_name"] == selected_lang)
        & year_condition
        & (book_idx2meta["average_rating"] >= min_rating)
    ]

    st.subheader(f"Top 5 Book Recommendations for User {selected_user}:")

shown_books = set()

BOOKS_SHOWN = 0
for i in recommended:
    book_id = list(book2idx.keys())[list(book2idx.values()).index(i)]
    if book_id in filtered_books.index and BOOKS_SHOWN < 5:
        row = filtered_books.loc[book_id]
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.image(row["small_image_url"] or DEFAULT_IMAGE, width=80)
            with cols[1]:
                st.markdown(f"📖 *{row['title']}*  \n✍️ _{row['authors']}_")
        BOOKS_SHOWN += 1
        shown_books.add(book_id)

if st.button("🎲 Surprise Me with 1 More Book"):
    for i in recommended:
        book_id = list(book2idx.keys())[list(book2idx.values()).index(i)]
        if book_id in filtered_books.index and book_id not in shown_books:
            row = filtered_books.loc[book_id]
            st.markdown("### 🎁 Surprise Recommendation!")
            st.image(row["small_image_url"] or DEFAULT_IMAGE, width=100)
            st.markdown(f"📖 **{row['title']}**  \n✍️ _{row['authors']}_")
            break

# === Footer ===
st.markdown(
    """---
Built with ❤️ using Neural Collaborative Filtering
"""
)
