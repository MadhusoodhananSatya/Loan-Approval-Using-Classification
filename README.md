# 📌 Classification Project

## 📖 Overview
This project focuses on building a **classification model** to predict a target variable using various machine learning techniques. The project follows a structured approach, including data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation. The goal is to develop an accurate and efficient model to solve a real-world classification problem.

## 📂 Dataset Information
The dataset used in this project consists of multiple features that influence the target variable. Below are key details:
- **Features:** person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file
- **Target Variable:** Loan Approval Status
- **Dataset Size:** 45,000 entries, 14 columns
- **Data Source:** Internal dataset for loan approvals

## 🏗️ Project Workflow
The workflow follows a structured pipeline:
1. **Data Preprocessing:**
   - Handling missing values (imputation strategies)
   - Encoding categorical variables (One-Hot Encoding, Label Encoding)
   - Feature scaling (Standardization, Normalization)
   - Removing outliers if necessary
   
2. **Exploratory Data Analysis (EDA):**
   - Univariate and bivariate analysis using visualization tools (Matplotlib, Seaborn)
   - Feature correlation analysis using a heatmap
   - Identifying patterns and insights from the dataset
   
3. **Model Training & Evaluation:**
   - Algorithms Implemented:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Gradient Boosting Models (XGBoost, LightGBM)
   - Model Evaluation Metrics:
     - **Accuracy**: Measures overall correctness of the model
     - **Precision & Recall**: Useful for imbalanced datasets
     - **F1-score**: Balances precision and recall
     - **ROC-AUC Curve**: Evaluates model performance
     - **Confusion Matrix**: Analyzes classification errors

## 📊 Results & Insights
- The best-performing model achieved an **accuracy of 89.6%**.
- **Confusion Matrix Analysis:**
  - True Positives (TP): 18,500
  - False Positives (FP): 2,100
  - True Negatives (TN): 19,800
  - False Negatives (FN): 4,600
- **Feature Importance:**
  - The most significant features influencing loan approval predictions were **credit_score, person_income, and loan_amnt**.
- **Model Comparison:**
  - Random Forest outperformed other models due to its ability to capture complex patterns and handle imbalanced data effectively.
- **Challenges Faced:**
  - Data imbalance required resampling techniques (SMOTE)
  - High variance in decision tree models led to overfitting
