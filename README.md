# Breast_Cancer_ML_classifications

Breast Cancer Classification
Overview
This project involves building and evaluating multiple machine learning models to classify breast cancer data. The goal is to identify the best-performing algorithm based on accuracy, precision, recall, and F1-score, using a preprocessed and feature-engineered dataset.

Dataset
Source: Scikit-Learn Data Set
The dataset contains data about breast cancer tumors, collected to help classify tumors as malignant or benign. It includes features that describe the properties of cell nuclei, derived from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Key Details
Source: The dataset is part of the UCI Machine Learning Repository.
Number of Samples: 569
Number of Features: 30 numerical features (real-valued).
Target Variable: Binary classification:
0: Malignant (cancerous)
1: Benign (non-cancerous)


Preprocessing Steps
Outlier Handling:
Outliers were capped with the mean to mitigate their influence on the model's performance.

Feature Selection:
The top 10 features were selected using the f_classif statistical test to reduce dimensionality and retain significant predictors.

Scaling:
Standard scaling was applied to ensure that all features have a mean of 0 and a standard deviation of 1, making the data suitable for algorithms sensitive to feature magnitudes.

Classification Algorithms Used
Logistic Regression
Random Forest
Decision Tree
Gradient Boosting
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)

Performance Metrics
The following metrics were used to evaluate the models:

Accuracy: Percentage of correct predictions.
Precision: Proportion of true positives among predicted positives.
Recall: Proportion of true positives among actual positives.
F1-Score: Harmonic mean of precision and recall.

Results Table
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.94	0.93	0.92	0.94
Random Forest	0.97	0.98	0.96	0.97
Decision Tree	0.93	0.93	0.93	0.93
Gradient Boosting	0.96	0.97	0.95	0.96
Support Vector Machine (SVM)	0.98	0.98	0.98	0.98
K-Nearest Neighbors (KNN)	0.93	0.92	0.91	0.92

Best Model:
Model: Support Vector Machine (SVM)
Performance: Achieved the highest accuracy (98.25%) and consistent performance across all metrics.

Worst Model:
Model: Decision Tree
Performance: Lowest accuracy (93.0%), though still reasonable for this dataset.

Visualization
Bar plots were created to compare model performance across metrics, making it easier to interpret the results.

Conclusion
SVM outperformed other models due to its ability to handle high-dimensional data and its sensitivity to feature scaling.
Decision Tree, while interpretable, suffered from overfitting and did not perform as well on unseen data.
