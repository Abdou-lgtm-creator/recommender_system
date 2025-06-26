"""
app.py

# This Streamlit app provides a personalized book recommendation interface using a trained Neural Collaborative Filtering (NCF) model.
# Users can select their ID to receive top book suggestions based on past ratings, with additional options for surprise picks and filters.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
    "nl": "Deutsch",  # Updated here
    "mul": "Multiple",
    "": "Unknown"
}

# === Load model and data ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/ncf_model.h5")

@st.cache_data
def load_data():
    ratings = pd.read_csv("../data/filtered_ratings.csv")
    books = pd.read_csv("../data/books.csv")
    user_map = pd.read_csv("../data/user_mapping.csv")
    book_map = pd.read_csv("../data/book_mapping.csv")

    # Filter invalid image URLs
    books = books[books['small_image_url'].str.startswith('http', na=False)]

    # Add readable language names
    books['language_name'] = books['language_code'].map(LANGUAGE_MAP).fillna(books['language_code'])

    return ratings, books, user_map, book_map

model = load_model()
ratings, books, user_map, book_map = load_data()

# === Mappings ===
user2idx = dict(zip(user_map['user_original'], user_map['user_index']))
book2idx = dict(zip(book_map['book_original'], book_map['book_index']))
book_idx2meta = books.set_index('book_id')[['title', 'authors', 'small_image_url', 'original_publication_year', 'average_rating', 'language_code', 'language_name']]

# === Styling ===
st.set_page_config(page_title="📚 Book Recommender", layout="centered")

st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        background-image: url('https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label, .stText, .stMarkdown, .stButton > button {
        color: white !important;
    }

    .stButton>button {
        background-color: #ffffff22;
        color: white !important;
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid #fff;
        font-weight: bold;
    }

    [data-baseweb="select"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 500;
    }

    [data-baseweb="menu"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
    }

    [data-baseweb="menu"] div[role="option"] {
        color: white !important;
    }

    [data-baseweb="menu"] div[role="option"]:hover {
        background-color: #444 !important;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# === App Header ===
st.markdown("""
    <div style='text-align: center;'>
        <h1>NextChapter</h1>
        <p style='font-size: 20px;'>Discover books tailored just for you 📖</p>
    </div>
""", unsafe_allow_html=True)

# === Select User ===
user_ids = list(user2idx.keys())
selected_user = st.selectbox("Select a User ID", user_ids)

# === Filter Options ===
st.sidebar.header("📂 Filter Options")

lang_options = books['language_name'].dropna().unique()
selected_lang = st.sidebar.selectbox("Language", sorted(lang_options))

min_year = 0
max_year = int(books['original_publication_year'].max())
selected_year_range = st.sidebar.slider("Publication Year", min_year, max_year, (2000, 2020))

min_rating = st.sidebar.slider("Minimum Average Rating", 1.0, 5.0, 3.5)

# Placeholder for missing image URLs
default_image = "https://via.placeholder.com/80x120.png?text=No+Image"

# === Generate Recommendations ===
if selected_user:
    user_idx = user2idx[selected_user]
    n_books = len(book2idx)
    book_indices = np.arange(n_books)
    user_array = np.full(n_books, user_idx)

    scores = model.predict([user_array, book_indices], verbose=0).flatten()

    rated_books = ratings[ratings['user_id'] == selected_user]['book_id'].map(
        book2idx).dropna().astype(int).values

    recommended = [i for i in np.argsort(scores)[::-1] if i not in rated_books]

    # Apply filters
    filtered_books = book_idx2meta[
        (book_idx2meta['language_name'] == selected_lang) &
        (book_idx2meta['original_publication_year'].between(*selected_year_range)) &
        (book_idx2meta['average_rating'] >= min_rating)
    ]

    st.subheader(f"Top 5 Book Recommendations for User {selected_user}:")

    shown = 0
    for i in recommended:
        book_id = list(book2idx.keys())[list(book2idx.values()).index(i)]
        if book_id in filtered_books.index and shown < 5:
            row = filtered_books.loc[book_id]
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(row['small_image_url'] or default_image, width=80)
                with cols[1]:
                    st.markdown(f"📖 **{row['title']}**  \n✍️ _{row['authors']}_")
            shown += 1

    if st.button("🎲 Surprise Me with 1 More Book"):
        for i in recommended:
            book_id = list(book2idx.keys())[list(book2idx.values()).index(i)]
            if book_id in filtered_books.index and book_id not in rated_books:
                row = filtered_books.loc[book_id]
                st.markdown("### 🎁 Surprise Recommendation!")
                st.image(row['small_image_url'] or default_image, width=100)
                st.markdown(f"📖 **{row['title']}**  \n✍️ _{row['authors']}_")
                break

# === Footer ===
st.markdown("""---  
Built with ❤️ using Neural Collaborative Filtering
""")
