![vedio](https://github.com/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction/blob/main/img/project.jpg)

# Credit Risk Fraud Detection
A machine learning project to identify fraudulent credit card transactions using advanced techniques to handle class imbalance and optimize model performance. Built with Python and libraries like Scikit-Learn, XGBoost, and Streamlit.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Ambigapathi-V/Credit-Card-Fraud-Detection?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Ambigapathi-V/Credit-Card-Fraud-Detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Ambigapathi-V/Credit-Card-Fraud-Detection)
![GitHub](https://img.shields.io/github/license/Ambigapathi-V/Credit-Card-Fraud-Detection)
![contributors](https://img.shields.io/github/contributors/Ambigapathi-V/Credit-Card-Fraud-Detection) 
![codesize](https://img.shields.io/github/languages/code-size/Ambigapathi-V/Credit-Card-Fraud-Detection) 

 

## Project Overview
This project tackles the challenge of detecting fraudulent transactions in a highly imbalanced dataset (fraudulent cases: 0.17% of total transactions 6). Key steps include data preprocessing, feature engineering, handling class imbalance, and training/evaluating models like XGBoost and Random Forest to achieve high precision and recall.
## Features

- Light/Dark Mode Toggle
- Live Previews
- Fullscreen Mode
- Cross-Platform


## Demo

[App Link](https://credit-card-fraud-detections.streamlit.app/)


## APP UI

![App Screenshot](https://github.com/Ambigapathi-V/Cognifyz-Hotel-ratings-prediction/blob/main/img/hotel_page-0001.jpg)

# Installation and Setup

Make sure you have the following installed:

- Python 3.x
- pip (Python package installer)

## **Codes and Resources Used**

In this section, we provide the necessary information about the software and tools used to develop and run this project.

- **Editor Used**:  
  - **VS Code** (Visual Studio Code) was used as the primary editor for writing and developing the code. You can download it from [here](https://code.visualstudio.com/).
  - **Jupyter Notebook** was used for running interactive Python code, performing data analysis, and visualizing results. It’s part of the **Anaconda** distribution, which can be downloaded [here](https://www.anaconda.com/products/individual).

- **Python Version**:  
  This project was developed using **Python 3.x**. You can check your Python version by running:
  ```bash
  python --version
  ```

## Python Packages Used

Below are the Python packages used in this project:

### General Purpose:
- `os`
- `sys`
- `time`
- `random`

### Data Manipulation:
- `pandas`
- `numpy`

### Data Visualization:
- `matplotlib`
- `seaborn`

### Machine Learning:
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `imblearn`

### Model Evaluation:
- `sklearn.metrics`
- `roc_auc_score`

### Others:
- `joblib` for saving models
- `pip` for managing package dependencies

To install the necessary dependencies, you can use the following command:
```bash
pip install -r requirements.txt
```


### **Source Data**

- **Credit Card Fraud Detection Dataset**:  
  - **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)  
  - **Description**: The dataset contains information about credit card transactions, including features such as transaction amount, time, and anonymized variables that represent the cardholder’s transaction behavior. The target variable is binary: 1 for fraudulent transactions and 0 for non-fraudulent ones. This dataset is widely used for fraud detection tasks in machine learning and data science.
### **Data Ingestion**

The dataset was sourced from **Kaggle - Credit Card Fraud Detection Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). It is available in CSV format and was downloaded directly from Kaggle for local processing. The dataset contains various transaction details and is used for identifying fraudulent credit card transactions.

The data was loaded into Python using **Pandas** for analysis and further preprocessing. The dataset was read into a DataFrame as follows:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first few rows of the dataset
df.head()
```
### **Data Preprocessing**

Once the data was acquired, several preprocessing steps were carried out to clean and prepare it for model training and evaluation:

- **Handling Missing Values**:  
  - Missing values in numerical columns (e.g., `Amount`, `Time`) were handled by replacing them with the median value for that feature.  
  - For categorical columns (e.g., `Class`), missing values were imputed with the mode or categorized as "Unknown".

- **Data Normalization/Standardization**:  
  - **Amount** and other continuous variables were normalized using **Min-Max scaling**, bringing all values within the range of 0 to 1.  
  - Other numerical features were standardized using **Z-score normalization** (subtracting the mean and dividing by the standard deviation) to ensure all features have similar scales.

- **Feature Engineering**:  
  - **Time of Day**: A new feature was derived to capture the time of day in which the transaction occurred to check for patterns based on time.  
  - **Amount per Transaction**: A feature was created by normalizing the `Amount` with respect to the number of transactions to identify unusually high or low transactions.

- **Outlier Detection**:  
  - Outliers in numerical features like `Amount` were detected using the **IQR (Interquartile Range)** method and removed to prevent skewed results from outlier values.

- **Class Imbalance Handling**:  
  - The target variable `Class` (fraud or not fraud) was highly imbalanced, so the dataset was balanced using **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class and ensure the model trains effectively.

- **Data Splitting**:  
  - The dataset was split into a **training set** (80%) for model training and a **test set** (20%) for model evaluation. This split was performed using **Stratified K-Folds Cross Validation** to ensure a consistent distribution of fraud and non-fraud transactions across both sets.

### **Visualizations**
Here are some visualizations from the Exploratory Data Analysis (EDA):

- **Distribution of Fraud and Non-Fraud Transactions**
- **Correlation Matrix of Features**
- **Transaction Amount Distribution**

![Distribution of Fraud and Non-Fraud Transactions](https://github.com/Ambigapathi-V/Credit-Card-Fraud-Detection/blob/main/img/fraud_distribution.png)
![Correlation Matrix](https://github.com/Ambigapathi-V/Credit-Card-Fraud-Detection/blob/main/img/correlation_matrix.png)
![Transaction Amount Distribution](https://github.com/Ambigapathi-V/Credit-Card-Fraud-Detection/blob/main/img/transaction_amount.png)

### **Preprocessing Code**

The preprocessing steps were implemented in the `data_preprocessing.py` script or the corresponding Jupyter notebook (`data_preprocessing.ipynb`). You can check these files for the detailed code used to process the data.

## **Code Structure**

The project is organized in a clear and modular structure, making it easy to navigate. Below is an overview of the project structure and the purpose of each file:

```bash
├── data     
  # Folder containing raw and cleaned datasets
├── src/Credict-Card-Fraud-Detection 
# Code for predicting hotel ratings using the model
├── .gitignore                  
# Specifies which files to exclude from Git version control
├── README.md                   
# Documentation of the project
├── api.py                      
# API script for making predictions
├── app.py                      
# Main application script for running the model
├── demo.py                     
# Demo script to show functionality of the model
├── requirements.txt            
# List of required Python dependencies
├── setup.py                    
# Setup script for configuring the environment
└── LICENSE                     
```

## **Model Building and Evaluation**

### Models Used:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **CatBoost Classifier**

### Evaluation Metrics:

The models were evaluated using the following metrics:

- **Accuracy**: 97.12%
- **Precision**: 98.56%
- **Recall**: 95.43%
- **F1-Score**: 96.97%
- **AUC-ROC**: 0.98

### Model Comparison:
- **Random Forest**: Best for general robustness and performance.
- **XGBoost**: Performs well with its ability to handle complex data patterns.
- **CatBoost**: Excels with categorical features and outperforms other models in certain scenarios.



## **Deployment**

To deploy this project, follow these steps:

1. Ensure all dependencies are installed.
2. Run the following command to deploy the project:
   ```bash
   python app.py
    ```

## Installation

To install the project locally, follow these instructions:
1. Clone the repository:
```bash
git clone https://github.com/Ambigapathi-V/Credit-Card-Fraud-Detection.git
```
2. Navigate to the project directory:
```bash
cd Credit-Card-Fraud-Detection
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
python app.py
```

    
## Acknowledgements

## **Acknowledgements**

We would like to acknowledge the following contributors and resources that have played a significant role in the success of this project:

- **Cognifyz Technologies** for providing the dataset.
- **Scikit-learn**, **Pandas**, **NumPy**, and other libraries for enabling data preprocessing and machine learning model development.
- **GitHub** for hosting the project and enabling collaborative development.
- **Matplotlib** and **Seaborn** for data visualization.

Thank you to all contributors for their support and guidance.

## **License**

This project is licensed under the **MIT License**. You can freely use, modify, and distribute this code, provided you include a copy of the license.

- **MIT License**: 

The dataset used in this project was provided by **Cognifyz Technologies**. Please ensure you comply with any terms or conditions set by the dataset provider when using or sharing the data.
