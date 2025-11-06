
# Comparing Classifiers: Predicting Term Deposit Subscriptions

## Assignment Notebook
*[(Link to notebook can be inserted here)](https://github.com/ojbskvasu/ML-Assignment-Comparing-Classifiers/blob/main/bank_classify.ipynb)*

## Problem Statement

The goal of this assignment is to compare the performance of various classification algorithms (K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines) on a bank marketing dataset. The primary business objective is to predict whether a client will subscribe to a term deposit after a marketing campaign to help a banking institution optimize its marketing efforts.

## Data Overview

**File Location:** data/bank-additional-full.csv

**Description:** The dataset contains information related to direct marketing campaigns (phone calls) of a Portuguese banking institution.

**Attributes:** The dataset includes 21 attributes covering bank client data, last contact information, other attributes (like campaign contacts and days since last contact), and social and economic context attributes. The target variable is 'y', indicating whether the client subscribed to a term deposit (binary: 'yes' or 'no'). The dataset contains 41,188 instances.

## Key Findings

Based on the analysis conducted in the notebook:

### Data Analysis and Preparation:

*   **Missing Values:** The dataset was found to have no missing values across all columns.
*   **Feature Engineering (Bank Data):** Selected bank client features (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`) were prepared for modeling by applying one-hot encoding to categorical variables and encoding the binary target variable ('y').
*   **Train/Test Split:** The prepared data was split into training and testing sets (75/25 split).

### Model Performance Comparison (Using Bank Client Features):

*   **Baseline Performance:** A baseline accuracy was established based on predicting the majority class in the training data (approximately 0.8871).
*   **Initial Model Comparison (Default Settings):**
    *   Logistic Regression and Support Vector Machines (SVM) achieved the highest test accuracy (0.8880), but SVM had a significantly longer training time compared to Logistic Regression.
    *   K Nearest Neighbor (KNN) had a test accuracy of 0.8791 and was the fastest to train.
    *   Decision Tree had the lowest test accuracy (0.8645) but a relatively fast training time.

*   **Hyperparameter Tuning:**
    *   Tuning the Decision Tree (`max_depth`, `min_samples_split`, `min_samples_leaf`) improved its test accuracy to 0.8865.
    *   Tuning the KNN (`n_neighbors`) improved its test accuracy to 0.8862.

*   **Overall Performance:** While tuning improved the Decision Tree and KNN, the default Logistic Regression and SVM models still achieved slightly higher accuracy on this feature set. Logistic Regression offered a good balance of accuracy and training efficiency.

## How to Run the Notebook

1.  Ensure you have the required libraries installed (pandas, scikit-learn, matplotlib).
2.  Make sure the `bank-additional-full.csv` dataset is accessible at `data/bank-additional-full.csv` relative to the notebook, or update the file path accordingly.
3.  Execute the cells sequentially to replicate the data loading, preprocessing, model training, evaluation, and tuning steps.
