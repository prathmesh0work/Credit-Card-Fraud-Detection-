# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using **Logistic Regression** and **Random Forest** classifiers, with SMOTE-based class imbalance handling.

---

## 📌 Problem Statement

Credit card fraud is a major financial threat. This project builds a binary classification model to distinguish between **legitimate (Class 0)** and **fraudulent (Class 1)** transactions using the well-known [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset is heavily imbalanced — fraudulent transactions account for a very small percentage of all records — making it a challenging and realistic classification task.

---

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── credit_card_fraud_detection.ipynb   # Main Jupyter notebook
├── README.md                           # Project documentation
└── creditcard.csv                      # Dataset (not included — see below)
```

---

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, data types, and summary statistics
- Missing value check
- Class distribution visualization (countplot)
- Correlation heatmap
- Feature correlation with the target variable (`Class`)
- Boxplots for key features (`V11`, `V17`)
- Histogram distribution of all features

### 2. Data Preprocessing
- Feature/target split (`X`, `y`)
- Train-test split (80/20, `random_state=42`)
- Feature scaling using `StandardScaler`

### 3. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on the training set
- Compared class distribution before and after resampling

### 4. Model Training & Evaluation

| Model               | Notes                                      |
|---------------------|--------------------------------------------|
| Logistic Regression | Trained on full SMOTE-resampled data        |
| Random Forest       | `n_estimators=20`, `max_depth=8`, subset of 50k samples |

Both models evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score & ROC Curve
- Custom probability threshold tuning (threshold = 0.8)

### 5. Feature Importance
- Top 10 most important features identified from the Random Forest model

---

## 🏆 Results

**Random Forest outperformed Logistic Regression** because it captures non-linear relationships in the data that Logistic Regression cannot model effectively.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | ML models, preprocessing, evaluation |
| Imbalanced-learn | SMOTE for class balancing |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prathmesh0work/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

3. **Download the dataset**

   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

4. **Run the notebook**
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 31 total — `Time`, `V1`–`V28` (PCA-transformed), `Amount`, and `Class`
- **Target:** `Class` — 0 = Legitimate (284,315 transactions), 1 = Fraud (492 transactions)
- **Imbalance:** Only **0.17%** of transactions are fraudulent
- **File size:** ~144 MB

> ⚠️ The dataset is not included in this repository due to its size (~144 MB). Please download it separately from Kaggle and place it in the project root as `creditcard.csv`.

---

## 📈 Key Visualizations

- Class distribution bar chart
- Correlation heatmap across all features
- Boxplots: `V11` and `V17` by class
- ROC Curve for Random Forest
- Feature importance bar chart (top 10)

---

## 🚀 Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Try XGBoost or LightGBM for better performance
- Use cross-validation for more robust evaluation
- Deploy the model as a REST API using Flask or FastAPI

---

## 🙋 Author

**Prathmesh Ingole**
- GitHub: [@prathmesh0work](https://github.com/prathmesh0work)
- LinkedIn: [Prathmesh Ingole](https://linkedin.com/in/prathmesh-ingole)

---

## 📄 License

This project is open-source and available under the github/prathmesh0work.
