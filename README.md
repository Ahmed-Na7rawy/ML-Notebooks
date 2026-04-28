# Machine Learning Notebooks

A collection of 10 applied data science notebooks exploring real-world datasets. Each notebook goes beyond model fitting — focusing on understanding the data, extracting insights, and communicating findings clearly.

Built with Python, Scikit-learn, Pandas, and Matplotlib.

---

## Notebooks

### Classification

| Notebook | What it does | Key Finding | Metrics | Dataset |
|----------|-------------|-------------|---------|---------|
| `breast_cancer_knn_classification.ipynb` | Diagnoses breast cancer from cell measurements using K-Nearest Neighbors with pipeline preprocessing | Concave points and radius were the strongest predictors of malignancy. | Accuracy: 96% | [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) |
| `diabetes_svm_prediction.ipynb` | Predicts diabetes risk from health metrics using Support Vector Classification with kernel tuning | Glucose level alone accounts for more predictive power than all other features combined. | Accuracy: 78% | [Kaggle — Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| `heart_disease_naive_bayes.ipynb` | Identifies heart disease presence from clinical data using Gaussian Naive Bayes | Chest pain type and maximum heart rate achieved are the two most discriminating features. | Accuracy: 85% | [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease) |
| `apple_quality_decision_tree.ipynb` | Classifies apple quality from physical properties using Decision Trees | Sweetness and juiciness form the clearest decision boundary for quality classification. | Accuracy: 89% | [Kaggle — Apple Quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) |
| `customer_churn_prediction.ipynb` | Predicts which bank customers will leave using Logistic Regression and Random Forest | Customers with 3+ products and inactive membership status churn at 3× the average rate. | Accuracy: 87% | [Kaggle — Bank Customer Churn](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) |
| `shopping_behavior_prediction.ipynb` | Forecasts customer purchasing patterns from shopping behavior data | Discount sensitivity varies dramatically by age group — younger shoppers are 2× more responsive. | Accuracy: 82% | [Kaggle — Customer Shopping Trends](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset) |

### Regression

| Notebook | What it does | Key Finding | Metrics | Dataset |
|----------|-------------|-------------|---------|---------|
| `fifa19_player_value_prediction.ipynb` | Estimates FIFA 19 player market values using Support Vector Regression | Overall rating and potential have a non-linear relationship with market value — top-tier players see exponential value growth. | R² Score: 0.81 | [Kaggle — FIFA 19](https://www.kaggle.com/datasets/karangadiya/fifa19) |
| `taxi_fare_prediction.ipynb` | Predicts taxi trip fares from route and trip metadata using Linear Regression and tree models | Trip distance is the dominant predictor, but time-of-day adds meaningful lift due to surge pricing patterns. | R² Score: 0.87 | [Kaggle — NYC Taxi Fare](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) |

### Anomaly Detection

| Notebook | What it does | Key Finding | Metrics | Dataset |
|----------|-------------|-------------|---------|---------|
| `credit_card_anomaly_detection.ipynb` | Detects fraudulent credit card transactions using One-Class SVM on highly imbalanced data | Fraudulent transactions cluster tightly in PCA-transformed feature space, making unsupervised detection viable despite extreme class imbalance (0.17%). | Precision: 91% | [Kaggle — Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

### Full Analysis Projects

| Notebook | What it does | Key Finding | Dataset |
|----------|-------------|-------------|---------|
| `Gaming_Academic_Performance.ipynb` | Full exploratory analysis investigating how gaming habits affect academic performance, with statistical testing and visualization | Moderate gaming (1–2 hrs/day) correlates with higher grades than both non-gamers and heavy gamers, suggesting a non-linear relationship. | [Kaggle — Student Gaming & Academic Performance](https://www.kaggle.com/datasets/mrsimple07/student-gaming-and-academic-performance) |

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
