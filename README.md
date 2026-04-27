# Machine Learning Notebooks

A collection of 10 applied machine learning notebooks covering classification, regression, anomaly detection, and data analysis. Each notebook walks through a real dataset end-to-end: loading, cleaning, exploring, modeling, and evaluating.

Built using Python, Scikit-learn, Pandas, and Matplotlib.

---

## Notebooks

### Classification

| Notebook | What it does | Metrics | Dataset |
|----------|-------------|---------|---------|
| `breast_cancer_knn_classification.ipynb` | Diagnoses breast cancer from cell measurements using K-Nearest Neighbors with pipeline preprocessing | Accuracy: 95% | [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) |
| `diabetes_svm_prediction.ipynb` | Predicts diabetes risk from health metrics using Support Vector Classification with kernel tuning | Accuracy: 78% | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| `heart_disease_naive_bayes.ipynb` | Identifies heart disease presence from clinical data using Gaussian Naive Bayes | Accuracy: 82% | [UCI](https://archive.ics.uci.edu/ml/datasets/heart+Disease) |
| `apple_quality_decision_tree.ipynb` | Classifies apple quality from physical properties using Decision Trees | Accuracy: 89% | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) |
| `customer_churn_prediction.ipynb` | Predicts which bank customers will leave using Logistic Regression and Random Forest | F1-Score: 0.84 | [Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) |
| `shopping_behavior_prediction.ipynb` | Forecasts customer purchasing patterns from shopping behavior data | Accuracy: 85% | [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset) |

### Regression

| Notebook | What it does | Metrics | Dataset |
|----------|-------------|---------|---------|
| `fifa19_player_value_prediction.ipynb` | Estimates FIFA 19 player market values using Support Vector Regression | R2: 0.91 | [Kaggle](https://www.kaggle.com/datasets/karangadiya/fifa19) |
| `taxi_fare_prediction.ipynb` | Predicts taxi trip fares from route and trip metadata using Linear Regression and tree models | RMSE: $3.50 | [Kaggle](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) |

### Anomaly Detection

| Notebook | What it does | Metrics | Dataset |
|----------|-------------|---------|---------|
| `credit_card_anomaly_detection.ipynb` | Detects fraudulent credit card transactions using One-Class SVM on highly imbalanced data | F1-Score: 0.81 | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

### Full Analysis Projects

| Notebook | What it does | Metrics | Dataset |
|----------|-------------|---------|---------|
| `Gaming_Academic_Performance.ipynb` | Full exploratory analysis investigating how gaming habits affect academic performance, with statistical testing and visualization | N/A | [Kaggle](https://www.kaggle.com/datasets/mrsimple07/student-gaming-and-academic-performance) |

---

## Tech Stack

- Python 3.x
- Scikit-learn — classification, regression, clustering
- Pandas and NumPy — data manipulation
- Matplotlib and Seaborn — visualization
- Jupyter Notebook

## What's Covered

- Supervised learning (classification and regression)
- Unsupervised learning (anomaly detection)
- Feature engineering and selection
- Model evaluation (confusion matrix, ROC-AUC, cross-validation)
- Hyperparameter tuning
- Data preprocessing pipelines

---

## Getting Started

```bash
git clone https://github.com/Ahmed-Na7rawy/ML-Notebooks.git
cd ML-Notebooks
pip install -r requirements.txt
jupyter notebook
```

---

Ahmed Alnahrawy — [@Ahmed-Na7rawy](https://github.com/Ahmed-Na7rawy)
