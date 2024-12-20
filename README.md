**Machine Learning-Based Loan Default Prediction**

This project uses **machine learning techniques** to predict loan defaults by analyzing borrower demographics, financial history, and loan characteristics. It compares the performance of 10 supervised learning models to determine the most effective method for identifying high-risk borrowers.

---

📜 **Overview**

- **Objective:** Predict whether a borrower will default on a loan using machine learning models.
- **Dataset:** Loan-related data with 255,347 records and multiple attributes, such as income, loan amount, credit score, and employment type.
- **Challenges:** Addressing class imbalance (88% non-defaults vs. 12% defaults) and selecting meaningful features.

---

✨ **Key Features**

1. **Data Preprocessing**
- **Class Imbalance Handling:** Used **SMOTE (Synthetic Minority Oversampling Technique)** to improve model performance.
- **Feature Scaling:** Normalized numerical variables for comparability.
- **Encoding:** Applied one-hot encoding for categorical variables and converted binary variables (e.g., Has Mortgage) to dummy variables.
- **Interaction Terms:** Explored interactions like `Income × Loan Amount` but removed them due to performance degradation.

2. **Exploratory Data Analysis**
- Visualized key patterns using:
  - **Density plots:** Showed that lower income and higher loan amounts are associated with defaults.
  - **Boxplots:** Highlighted the relationship between credit score, loan amount, and default likelihood.
  - **Bar charts:** Compared default rates across demographics and loan purposes.

3. **Modeling Techniques**
Implemented 10 supervised learning models:
1. **Penalized Logistic Regression (AUC: 0.6948):** Baseline model, struggled with non-linear relationships.
2. **Random Forest (AUC: 0.8081):** Best performer; excelled at capturing complex feature interactions.
3. **XGBoost (AUC: 0.7911):** Close competitor to Random Forest; effective on imbalanced datasets.
4. **Neural Networks (AUC: 0.6944):** Captured complex patterns but required extensive tuning.
5. **Support Vector Machine (AUC: 0.7144):** Effective for simple features; used a radial kernel.
6. **Gradient Boosting Machine (AUC: 0.7857):** Competitive performance; required careful tuning.
7. **Decision Tree (AUC: 0.6562):** Easy to interpret but prone to overfitting.
8. **Elastic Net Regression (AUC: 0.6948):** Combined LASSO and Ridge penalties for multicollinearity handling.
9. **Naive Bayes (AUC: 0.7697):** Simple yet efficient, especially with preprocessed data.
10. **K-Nearest Neighbors (AUC: 0.6986):** Effective for simple patterns but computationally expensive.

4. **Performance Evaluation**
- **AUC-ROC Curves:** Evaluated each model's ability to differentiate between defaulters and non-defaulters.
- **Bar Plot Comparison:** Visualized AUC values for all models, highlighting the superior performance of Random Forest and XGBoost.

---

📊 **Results**

| Model                          | AUC-ROC | Key Observations                                              |
|--------------------------------|---------|---------------------------------------------------------------|
| **Penalized Logistic Regression** | 0.6948  | Baseline model; limited by non-linear patterns.               |
| **Random Forest**              | 0.8081  | Best overall performer; robust and interpretable.             |
| **XGBoost**                    | 0.7911  | Close competitor; excels on imbalanced datasets.              |
| **Neural Networks**            | 0.6944  | Moderate performance; computationally heavy.                  |
| **Support Vector Machine**     | 0.7144  | Moderate performance with smooth decision boundaries.         |
| **Gradient Boosting Machine**  | 0.7857  | Competitive; required careful hyperparameter tuning.          |
| **Decision Tree**              | 0.6562  | Simple and interpretable but less accurate.                   |
| **Elastic Net Regression**     | 0.6948  | Effective for multicollinearity; comparable to Logistic Regression. |
| **Naive Bayes**                | 0.7697  | Efficient for simpler problems; surprising performance.        |
| **K-Nearest Neighbors**        | 0.6986  | Moderate performance; best for identifying simple patterns.   |

---

🚀 **How to Run the Project**

Prerequisites
- **R (3.6 or later)** and the following packages:
  ```R
  install.packages(c(
    "caret", "randomForest", "xgboost", "nnet", "e1071", 
    "DMwR", "glmnet", "ggplot2", "rpart", "gbm", "doParallel"
  ))


**Steps**

1. **Clone the Repository:**
> git clone https://github.com/Singh050/Machine-Learning-Based-Prediction-of-Loan-Default.git
> cd Machine-Learning-Based-Prediction-of-Loan-Default

2. **Run the R Script: Open RStudio or an R terminal and execute:**
> source("Project_Final_Code_IDA.R")

3. **Visualizations and Outputs:**
> Density plots, boxplots, AUC-ROC curves, and bar charts for model performance will be generated.

**Key Visualizations**
1. ROC Curves: Compare sensitivity and specificity trade-offs for each model.
2. Bar Plot: Visualize AUC values for all models to highlight top performers.
3. Density and Boxplots: Analyze relationships between income, loan amount, and default risk.


**Insights**

> **Top Predictor:** Income emerged as the strongest predictor, followed by credit score.

> **Best Model:** Random Forest achieved the highest AUC-ROC (0.8081), making it the most reliable for deployment.

> **Effectiveness of SMOTE:** Balanced the dataset and improved detection of minority class (defaults).


**Contact**

**For collaboration or questions:**

**Barjinder Singh** - barjindersingh@ou.edu 


---

This code is ready for direct copy-paste into your repository’s `README.md` file. Let me know if you need further adjustments!
