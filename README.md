# 📚 NextChapter: LLM-Enhanced Book Recommendation System

A personalized book recommender using **Neural Collaborative Filtering (NCF)** enhanced with **Large Language Model (LLM) embeddings** for semantically rich recommendations.

---

## Features

- Personalized top-N book recommendations
- Uses **OpenAI or SentenceTransformers embeddings** for book metadata
- Neural Collaborative Filtering using TensorFlow/Keras
- Streamlit web interface for interactive recommendations
- Filtering by language, publication year, and rating
- Evaluation with RMSE and MAE metrics with loss/metric plots
- Modular pipeline:
    • Data preprocessing
    • LLM embeddings generation
    • Model training
    • Recommendation delivery

---

## 📁 Project Structure

```text
recommender_system/
├── app/
│   └── app.py                 # Streamlit frontend for recommendations
├── data/                      # Raw and processed data and LLM embeddings
├── models/                    # Saved trained model
├── src/
│   ├── results/               # Evaluation results
│   ├── preprocessing.py       # Data preprocessing
│   ├── recommend.py           # CLI recommendations
│   ├── train.py               # Model training
│   └── llm_embeddings.py      # Generate LLM embeddings for books
├── environment.yml            # Conda environment configuration
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```
---

## 🔧 Environment Setup

### Prerequisites
- Python 3.10+
- Conda or Miniconda installed

### Installation

1️⃣ Clone the repository:
    git clone https://github.com/your_username/recommender_system.git
    cd recommender_system

2️⃣ Create conda environment:
    conda env create -f environment.yml

3️⃣ Activate the environment:
    conda activate recommender-env

4️⃣ Verify installation:
    python --version
    pip list

---

## Usage

### Running the Application

1️⃣ Activate the environment:
    conda activate recommender-env

2️⃣ Navigate to the project directory:
    cd recommender_system

3️⃣ Run the Streamlit app:
    streamlit run app/app.py

---

### Development Pipeline

1️⃣ Data preprocessing:
    python src/preprocessing.py

2️⃣ Generate LLM embeddings:
    python src/llm_embeddings.py

3️⃣ Model training:
    python src/train.py

4️⃣ Generate recommendations:
    python src/recommend.py

---

## 📊 Evaluation

- Visualize loss and metric plots in `src/results/`
- Evaluate using RMSE and MAE on test data

---

## 🛠️ Development

### Module Structure

- `preprocessing.py`: Data cleaning and transformation
- `llm_embeddings.py`: Generate semantic embeddings using LLMs
- `train.py`: Train the LLM-enhanced NCF model
- `recommend.py`: CLI interface for recommendations
- `app.py`: Streamlit web app for interactive use

---

### Deactivating Environment

    conda deactivate

### Removing Environment (if needed)

    conda env remove -n recommender-env

---
