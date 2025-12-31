# DVC-S3-MLOps-Project

This project demonstrates an end-to-end Machine Learning pipeline for spam detection using DVC (Data Version Control) for experiment tracking and data versioning with AWS S3 integration.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Setup](#project-setup)
- [Pipeline Stages](#pipeline-stages)
- [DVC Pipeline Setup](#dvc-pipeline-setup)
- [Experiment Tracking](#experiment-tracking)
- [Usage](#usage)
- [Project Structure Details](#project-structure-details)

## 🎯 Overview

This project covers the complete understanding of creating an ML pipeline and working with DVC for:
- **Data Versioning**: Track and version your datasets
- **Experiment Tracking**: Log metrics and parameters for each experiment
- **Pipeline Automation**: Automate the ML workflow from data ingestion to model evaluation
- **Reproducibility**: Ensure experiments can be reproduced with exact parameters

## 📁 Project Structure

```
DVC-S3-MLOps-Project/
├── src/                      # Source code for pipeline stages
│   ├── data_ingestion.py     # Data loading and splitting
│   ├── data_preprocessing.py # Text preprocessing and cleaning
│   ├── feature_engineering.py # TF-IDF vectorization
│   ├── model_building.py     # Model training
│   └── model_evaluation.py   # Model evaluation and metrics
├── data/                     # Data directories (gitignored)
│   ├── raw/                  # Raw data
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final processed data
├── models/                   # Trained models (gitignored)
├── reports/                  # Evaluation reports
├── logs/                    # Log files
├── experiments/             # Jupyter notebooks for experimentation
├── dvclive/                 # DVC experiment tracking data
├── images/                  # Project images and diagrams
├── dvc.yaml                 # DVC pipeline configuration
├── params.yaml              # Hyperparameters configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## ✨ Features

- **Automated ML Pipeline**: Complete pipeline from data ingestion to model evaluation
- **DVC Integration**: Version control for data, models, and experiments
- **Parameter Management**: Centralized parameter configuration via `params.yaml`
- **Experiment Tracking**: Track metrics and parameters using dvclive
- **Text Preprocessing**: NLTK-based text cleaning and transformation
- **TF-IDF Vectorization**: Feature engineering for text classification
- **Random Forest Classifier**: ML model for spam detection

## 🔧 Prerequisites

- Python 3.8+
- Git
- DVC
- AWS Account (for S3 integration, optional)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd DVC-S3-MLOps-Project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### 4. Download NLTK Data

The NLTK data will be automatically downloaded when you run the preprocessing stage. Alternatively, you can download it manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🚀 Project Setup

### Step 1: Initial Setup

1. Create a GitHub repository and clone it locally
2. Add the `src` folder with all components
3. Test each component individually to ensure they work
4. Add `data/`, `models/`, `reports/` directories to `.gitignore`
5. Commit and push to GitHub

### Step 2: DVC Initialization

```bash
# Initialize DVC repository
dvc init

# Add remote storage (S3 example)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage

# Commit DVC configuration
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## 🔄 Pipeline Stages

The ML pipeline consists of 5 stages defined in `dvc.yaml`:

### 1. Data Ingestion (`data_ingestion.py`)
- Loads spam dataset from remote URL
- Preprocesses column names
- Splits data into train and test sets
- Saves raw data to `data/raw/`

**Parameters:**
- `test_size`: Test split ratio (default: 0.20)

### 2. Data Preprocessing (`data_preprocessing.py`)
- Encodes target labels
- Removes duplicates
- Applies text transformation:
  - Lowercase conversion
  - Tokenization
  - Stopword removal
  - Stemming using Porter Stemmer
- Saves processed data to `data/interim/`

### 3. Feature Engineering (`feature_engineering.py`)
- Applies TF-IDF vectorization
- Transforms text data into numerical features
- Saves feature-engineered data to `data/processed/`

**Parameters:**
- `max_features`: Maximum number of features for TF-IDF (default: 50)

### 4. Model Building (`model_building.py`)
- Trains Random Forest Classifier
- Saves trained model to `models/model.pkl`

**Parameters:**
- `n_estimators`: Number of trees in the forest (default: 25)
- `random_state`: Random seed for reproducibility (default: 2)

### 5. Model Evaluation (`model_evaluation.py`)
- Evaluates model performance
- Calculates metrics: accuracy, precision, recall, AUC
- Logs metrics using dvclive
- Saves metrics to `reports/metrics.json`

## 🔧 DVC Pipeline Setup

### Without Parameters

1. Create `dvc.yaml` file with pipeline stages
2. Test the pipeline:
   ```bash
   dvc repro
   ```
3. View pipeline DAG:
   ```bash
   dvc dag
   ```
4. Commit changes:
   ```bash
   git add dvc.yaml
   git commit -m "Add DVC pipeline"
   git push
   ```

### With Parameters

1. Create `params.yaml` file with hyperparameters
2. Update source code to load parameters:
   ```python
   import yaml
   
   def load_params(params_path: str) -> dict:
       with open(params_path, 'r') as file:
           params = yaml.safe_load(file)
       return params
   
   # In main function
   params = load_params('params.yaml')
   test_size = params['data_ingestion']['test_size']
   ```
3. Test the pipeline:
   ```bash
   dvc repro
   ```
4. Commit changes:
   ```bash
   git add params.yaml dvc.yaml src/
   git commit -m "Add parameterized pipeline"
   git push
   ```

## 📊 Experiment Tracking

### Setup dvclive

1. Install dvclive (already in requirements.txt):
   ```bash
   pip install dvclive
   ```

2. Add dvclive code to `model_evaluation.py`:
   ```python
   from dvclive import Live
   from sklearn.metrics import accuracy_score, precision_score, recall_score
   
   # In main function
   with Live(save_dvc_exp=True) as live:
       live.log_metric('accuracy', accuracy_score(y_test, y_pred))
       live.log_metric('precision', precision_score(y_test, y_pred))
       live.log_metric('recall', recall_score(y_test, y_pred))
       live.log_params(params)
   ```

### Running Experiments

1. Run an experiment:
   ```bash
   dvc exp run
   ```

2. View experiments:
   ```bash
   dvc exp show
   ```
   Or use the DVC extension in VS Code

3. Apply a previous experiment:
   ```bash
   dvc exp apply <experiment-name>
   ```

4. Remove an experiment:
   ```bash
   dvc exp remove <experiment-name>
   ```

5. Modify parameters in `params.yaml` and run new experiments:
   ```bash
   # Edit params.yaml
   dvc exp run
   dvc exp show
   ```

## 💻 Usage

### Run Complete Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# Run the entire pipeline
dvc repro

# Or run individual stages
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro feature_engineering
dvc repro model_building
dvc repro model_evaluation
```

### Modify Parameters

Edit `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.30

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 25
  random_state: 2
```

Then run:
```bash
dvc repro
```

### View Pipeline DAG

```bash
dvc dag
```

### Push Data to Remote Storage

```bash
# Push data to S3
dvc push

# Pull data from S3
dvc pull
```

## 📂 Project Structure Details

- **`src/`**: Contains all pipeline stage scripts
- **`data/raw/`**: Raw data files (train.csv, test.csv)
- **`data/interim/`**: Intermediate processed data
- **`data/processed/`**: Final feature-engineered data
- **`models/`**: Trained model files (.pkl)
- **`reports/`**: Evaluation metrics and reports
- **`logs/`**: Log files for each pipeline stage
- **`experiments/`**: Jupyter notebooks for experimentation
- **`dvclive/`**: DVC experiment tracking data
  - `metrics.json`: Experiment metrics
  - `params.yaml`: Experiment parameters
  - `plots/`: Metric plots
- **`images/`**: Project images, diagrams, and visualizations

## 📝 Parameters Configuration

The `params.yaml` file contains all hyperparameters:

```yaml
data_ingestion:
  test_size: 0.30

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 25
  random_state: 2
```

## 🔍 Monitoring and Logs

Each pipeline stage generates logs in the `logs/` directory:
- `data_ingestion.log`
- `data_preprocessing.log`
- `feature_engineering.log`
- `model_building.log`
- `model_evaluation.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the pipeline
5. Submit a pull request

## 📄 License

See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project demonstrates MLOps best practices using DVC for data versioning and experiment tracking.

---

**Note**: Make sure to add `data/`, `models/`, `reports/`, `logs/`, and `dvclive/` to `.gitignore` to avoid committing large files to Git. Use DVC to version control these files instead.
