# Employee Performance Rating Prediction using Machine Learning

## Project Overview

Employee performance evaluation plays a crucial role in workforce management and organizational growth. Accurate performance assessment helps organizations identify high-performing employees, improve productivity, and make informed HR decisions.

This project uses machine learning techniques to predict employee performance ratings based on employee demographics, job roles, work environment factors, and organizational attributes.

The solution assists HR departments in understanding key performance drivers and supports data-driven talent management strategies.

---

## Problem Statement

Traditional employee performance evaluations can be subjective and time-consuming. Organizations need a data-driven approach to identify performance trends and predict employee ratings accurately.

The objective of this project is to:

* Predict employee performance ratings.
* Identify factors influencing employee performance.
* Compare multiple machine learning models.
* Support HR decision-making with predictive analytics.

---

## Dataset Information

The dataset contains employee-related information collected from various departments.

### Features Included

| Feature Category     | Examples                              |
| -------------------- | ------------------------------------- |
| Employee Information | Age, Gender, Marital Status           |
| Job Details          | Department, Job Role, Business Travel |
| Education            | Education Level, Education Background |
| Work Environment     | OverTime, Environment Satisfaction    |
| Compensation         | Monthly Income, Percent Salary Hike   |
| Experience           | Total Working Years, Years at Company |
| Performance Metrics  | Training Times, Work-Life Balance     |
| Target Variable      | Performance Rating                    |

### Target Variable

| Rating | Description         |
| ------ | ------------------- |
| 2      | Low Performance     |
| 3      | Average Performance |
| 4      | High Performance    |

---

## Project Workflow

```text
Data Collection
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Outlier Treatment
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Model Deployment
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* XGBoost (Imported for experimentation)

---

## Exploratory Data Analysis (EDA)

Several analyses were conducted to understand employee characteristics and performance patterns.

### Data Quality Checks

* Dataset shape analysis
* Missing value detection
* Data type verification
* Statistical summary generation

### Visualizations Performed

* Numerical feature distributions
* Department-wise performance ratings
* Employee count by department
* Performance rating distribution
* Boxplots for outlier detection
* Correlation heatmap
* Performance analysis across categorical variables

### Key Findings

* Employee performance varies significantly across departments.
* Overtime and travel frequency show influence on ratings.
* Some numerical variables contain outliers.
* The dataset contains both categorical and numerical features requiring preprocessing.

---

## Data Preprocessing

### Feature Selection

The following steps were performed:

* Removed employee identifier column (`EmpNumber`)
* Separated features and target variable
* Handled numerical and categorical features independently

### Outlier Treatment

Outliers were capped using the IQR (Interquartile Range) method.

```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
```

### Encoding Techniques

#### Ordinal Encoding

Applied to:

* BusinessTravelFrequency

```python
OrdinalEncoder()
```

#### One-Hot Encoding

Applied to:

* Gender
* Marital Status
* Department
* Job Role
* Education Background
* OverTime

```python
OneHotEncoder()
```

### Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

### Preprocessing Pipeline

A complete preprocessing pipeline was built using:

```python
Pipeline()
ColumnTransformer()
```

This ensures consistency during training and deployment.

---

## Machine Learning Models Implemented

### 1. Logistic Regression

* Baseline classification model
* Fast training and prediction
* Easy interpretation

### 2. Decision Tree Classifier

* Handles nonlinear relationships
* Easy to visualize
* Provides feature-based decision rules

### 3. Random Forest Classifier

* Ensemble learning model
* Reduces overfitting
* Improves prediction accuracy
* Handles mixed feature types effectively

---

## Model Evaluation

Models were evaluated using:

* Accuracy Score
* Classification Performance Comparison

### Models Compared

| Model               | Purpose                  |
| ------------------- | ------------------------ |
| Logistic Regression | Baseline Model           |
| Decision Tree       | Nonlinear Classification |
| Random Forest       | Ensemble Learning        |

---

## Results

### Best Performing Model

🏆 **Random Forest Classifier**

Reasons:

* Highest prediction accuracy among tested models.
* Better generalization capability.
* Reduced overfitting compared to Decision Trees.
* Suitable for deployment in HR analytics systems.

---

## Feature Importance Insights

Key factors affecting employee performance include:

* Environment Satisfaction
* Job Role
* Department
* Years at Company
* Monthly Income
* Overtime
* Business Travel Frequency
* Work-Life Balance
* Training Opportunities

These insights can help HR teams improve employee engagement and productivity.

---

## Model Deployment

The trained model and preprocessing pipeline were saved using Joblib.

### Saved Files

```python
joblib.dump(RFC, "Employee_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
```

These files can be directly used in:

* Streamlit Applications
* Flask APIs
* FastAPI Deployments
* HR Analytics Dashboards

---

## Business Impact

This project can help organizations:

* Predict employee performance proactively.
* Identify high-performing employees.
* Improve promotion and appraisal processes.
* Design targeted training programs.
* Reduce performance management bias.
* Enhance workforce planning.

---

## Challenges Faced

* Handling numerous categorical variables.
* Managing class imbalance in performance ratings.
* Treating outliers without losing valuable information.
* Selecting the most effective machine learning model.

---

## Future Enhancements

* Hyperparameter tuning using GridSearchCV.
* Feature importance visualization.
* SHAP explainability integration.
* XGBoost and LightGBM implementation.
* Real-time HR analytics dashboard.
* Cloud deployment using Streamlit Cloud.

---

## Repository Structure

```text
Employee-Performance-Prediction/
│
├── data/
│   └── employee_performance.csv
│
├── notebooks/
│   └── Employee_Performance_Prediction.ipynb
│
├── models/
│   ├── Employee_model.pkl
│   └── preprocessor.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── assets/

## Sample Prediction Input

```text
Age: 30
Department: Sales
Job Role: Sales Executive
Business Travel: Travel_Rarely
Gender: Male
Marital Status: Married
Monthly Income: 5000
OverTime: Yes
Years At Company: 5
```

### Prediction Output

```text
Predicted Performance Rating: 4 ⭐
```

---

## Author

**Purandhar**

Machine Learning Project – Employee Performance Rating Prediction System

This README is polished for GitHub portfolios, internship applications, LinkedIn showcases, and HR analytics/data science project demonstrations.
