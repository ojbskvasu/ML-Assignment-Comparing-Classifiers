
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

## Findings and Recommendations

Based on the exploration and model comparison performed in this notebook using the "bank client data" features, here are the key findings and recommendations for the banking institution:

### Findings:

1.  **Data Quality:** The dataset is clean with no missing values, which simplifies the initial preprocessing steps.
2.  **Baseline Performance:** A simple baseline of predicting the majority class achieves an accuracy of approximately 88.71%, highlighting the need for models to significantly outperform this to be valuable.
3.  **Initial Model Performance:**
    *   Both **Logistic Regression** and **SVM** models with default settings achieved the highest test accuracy (0.8880), just slightly above the baseline.
    *   **Logistic Regression** was significantly faster to train than SVM, making it more practical for initial model development and potentially for larger datasets.
    *   **Decision Tree** and **KNN** models with default settings had lower accuracy (0.8645 and 0.8791 respectively).
4.  **Impact of Tuning:** Hyperparameter tuning improved the performance of the Decision Tree (to 0.8865) and KNN (to 0.8862) models, bringing their accuracy closer to Logistic Regression and SVM, but did not surpass them on this feature subset.
5.  **Feature Limitation:** The analysis so far has only used a subset of the available features ("bank client data"). Other features related to contact, and social/economic context are likely to be highly relevant for predicting term deposit subscriptions.

### Recommendations:

1.  **Consider Logistic Regression:** Given its competitive accuracy and significantly faster training time compared to SVM, the **Logistic Regression** model with default settings is a strong candidate for a production model based on the "bank client data" features alone.
2.  **Explore Additional Features:** The analysis should be extended to include the **contact, social, and economic context attributes**. These features are likely to provide valuable information and could significantly improve model performance across all algorithms.
3.  **Re-evaluate Models with Full Feature Set:** After incorporating additional features, re-train and compare all four classifiers (Logistic Regression, KNN, Decision Tree, and SVM) again. Hyperparameter tuning should also be revisited with the expanded feature set.
4.  **Consider Other Metrics:** While accuracy is a good starting point, evaluate models using metrics more aligned with the business objective, such as **Precision** (minimizing false positives - contacting clients unlikely to subscribe) and **Recall** (minimizing false negatives - missing out on clients who would subscribe). The F1-score provides a balance between precision and recall.
5.  **Operationalize the Best Model:** Once a satisfactory model is identified (considering performance, training time, and interpretability), it can be used to score new clients and prioritize marketing efforts towards those most likely to subscribe.

By following these recommendations, the banking institution can build a more effective predictive model to optimize their term deposit marketing campaigns.
