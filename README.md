# Recommender System - Book Recommendation System

An intelligent recommendation system designed to suggest personalized book recommendations to users based on their preferences and reading history.

## 📁 Project Structure

```
recommender_system/
├── app/
│   └── app.py                 # Main application (user interface)
├── data/                      # Raw and processed data
├── models/                    # Saved trained models
├── src/
│   ├── results/              # Evaluation results
│   ├── preprocessing.py      # Data preprocessing
│   ├── recommend.py          # Recommendation algorithms
│   └── train.py              # Model training
├── environment.yml           # Conda environment configuration
├── .gitignore               # Files to ignore by Git
└── README.md                # Project documentation
```

## 🔧 Environment Setup

### Prerequisites
- Python 3.8+
- Conda or Miniconda installed

### Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Abdou-lgtm-creator/recommender_system.git>
   cd recommender_system
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate recommender-env
   ```

4. **Verify installation**
   ```bash
   python --version
   pip list
   ```

## 🚀 Usage

### Running the Application

1. **Activate the environment** (if not already done)
   ```bash
   conda activate recommender-env
   ```

2. **Navigate to project directory**
   ```bash
   cd recommender_system
   ```

3. **Run the application**
   ```bash
   python app/app.py
   ```

### Development Pipeline

1. **Data preprocessing**
   ```bash
   python src/preprocessing.py
   ```

2. **Model training**
   ```bash
   python src/train.py
   ```

3. **Generate recommendations**
   ```bash
   python src/recommend.py
   ```

## 📊 Features

- **User Interface**: Intuitive web application with Streamlit for recommendations

## 🛠️ Development

### Module Structure

- **`preprocessing.py`**: Data cleaning and transformation
- **`train.py`**: Training and validation of recommendation models
- **`recommend.py`**: Generation of user recommendations
- **`app.py`**: Web interface to interact with the system

### Deactivating Environment

```bash
conda deactivate
```

### Removing Environment (if needed)

```bash
conda env remove -n recommender-env
```