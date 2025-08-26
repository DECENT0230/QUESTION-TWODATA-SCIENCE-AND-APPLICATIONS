##Fraud Detection in Banking Transactions
#Project Overview
This project implements a comprehensive machine learning system for detecting fraudulent banking transactions. Using a hybrid approach that combines both supervised and unsupervised learning techniques, the system effectively identifies suspicious patterns while minimizing false positives that could inconvenience legitimate customers.
 Problem Statement
Banks face significant challenges in detecting fraudulent transactions among millions of daily operations. The key difficulties include:

Extreme class imbalance (typically <1% fraud cases)

Constantly evolving fraud patterns

Need for real-time detection with low false positive rates

Mix of labeled and unlabeled transaction data
 Dataset Description
The dataset contains 200 synthetic banking transactions with the following features:

#Feature	Description	Type
Amount	Transaction amount in USD	Numerical
Time_Hour	Hour of day (0-23)	Numerical
Location	Transaction location (Online, ATM, Store)	Categorical
Merchant	Merchant category (Bank, Food, Grocery, Retail, Travel)	Categorical
Is_Fraud	Label (1=Fraud, 0=Legitimate, -1=Unlabeled)	Target
Class Distribution:

Legitimate transactions: 95 (47.5%)

Fraudulent transactions: 5 (2.5%)

Unlabeled transactions: 100 (50.0%)

#Technical Architecture
Data Processing Pipeline
text
Raw Data → Cleaning → Feature Engineering → Scaling → Model Training → Evaluation
Algorithm Selection
Unsupervised Learning: K-Means Clustering for pattern discovery

Supervised Learning: Naïve Bayes for classification

Class Imbalance Handling: ADASYN oversampling technique

#Implementation Details
i)K-Means Clustering (Unsupervised Learning)
Algorithm Justification:
K-Means was selected for its effectiveness in discovering natural groupings within unlabeled data. Its distance-based approach makes it suitable for identifying anomalous transactions that deviate significantly from cluster centroids.

#Key Steps:

One-Hot Encoding for categorical features (Location, Merchant)

StandardScaler normalization for numerical features

Elbow method to determine optimal k=4 clusters

PCA visualization for cluster interpretation

##Results:

Identified 4 distinct transaction patterns

Cluster 0 showed highest fraud rate (8.70%)

Successfully detected anomalous transaction groupings

ii) Naïve Bayes Classification (Supervised Learning)
Algorithm Selection Rationale:
Naïve Bayes was chosen for its:

Strong performance with imbalanced datasets

Probabilistic foundation suitable for categorical features

Computational efficiency for real-time applications

Low variance reducing overfitting risk

Bias-Variance Tradeoff:
The algorithm maintains high bias (due to feature independence assumptions) but low variance, making it robust against overfitting—a crucial characteristic for fraud detection with limited positive examples.

iii) Feature Engineering Strategy
Created Features:

Temporal Features:

Is_Night (10 PM - 6 AM)

Is_Business_Hours (9 AM - 5 PM)

Amount Transformations:

Amount_Log (logarithmic scaling)

High_Amount (top 10% transactions)

Interaction Features:

ATM_Night_High_Amount (high-risk combination)

Class Imbalance Handling:

ADASYN oversampling: Generated synthetic minority samples

Strategic feature creation: Enhanced discriminative power without relying heavily on rare fraud examples

Variance reduction: Engineered features created more stable decision boundaries

iv) Comprehensive Evaluation
Metrics Used:

F1-Score: Primary metric (balances precision and recall)

10-Fold Cross-Validation: Robust performance assessment

Confusion Matrix: Detailed error analysis

ROC & Precision-Recall Curves: Comprehensive visual evaluation

Performance Results:

F1-Score: 0.8889 (exceeds target of 0.8)

Cross-validated F1: 0.9039 ± 0.0306

Precision: 0.82, Recall: 0.97

High-risk detection: 23 transactions identified (p > 0.7)

##Results Analysis
Supervised Learning Performance
The Naïve Bayes classifier demonstrated excellent performance with:

High recall (97%) ensuring most fraud cases are detected

Good precision (82%) minimizing false positives

Strong generalization (cross-validation F1: 0.9039)

Unsupervised Learning Insights
K-Means clustering revealed:

Distinct transaction patterns based on amount, time, and location

Cluster 0 emerged as highest risk (8.70% fraud rate)

Useful for discovering novel fraud patterns beyond known labels

Feature Importance
The most predictive features were:

Transaction Amount

Time of Day

Merchant Type (Bank transactions higher risk)

Location (ATM transactions higher risk)

## Deployment Recommendations
Real-Time Implementation
Transaction Scoring: Deploy Naïve Bayes model for real-time probability scoring

Threshold Tuning: Use 0.5 threshold for classification, 0.7+ for urgent review

Hybrid Approach: Combine supervised scoring with unsupervised pattern detection

Monitoring and Maintenance
Regular Retraining: Update model with new labeled data monthly

Performance Tracking: Monitor precision, recall, and F1-score continuously

Pattern Analysis: Use clustering to detect emerging fraud patterns

Risk Management
Prioritization: Focus investigations on high-probability cases first

Customer Experience: Balance fraud detection with minimal false positives

Adaptive Thresholds: Adjust sensitivity based on transaction context

##Future Enhancements
Advanced Algorithms: Experiment with ensemble methods and deep learning

Additional Features: Incorporate user behavior patterns and historical data

Real-time API: Develop REST API for integration with banking systems

Dashboard: Create monitoring dashboard with visual analytics

Adaptive Learning: Implement continuous learning from new fraud cases

##Requirements
bash
Python 3.8+
scikit-learn==1.2.2
imbalanced-learn==0.10.1
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
##Academic Compliance
This solution addresses all assignment requirements:

-K-Means implementation with elbow method and visualization
-Naïve Bayes classification with bias-variance explanation
-Feature engineering for imbalance handling
- Cross-validation with F1-score > 0.8
- Supervised vs unsupervised performance comparison

##Key Insights
Hybrid Approach: Combining supervised and unsupervised learning provides comprehensive coverage

Feature Engineering: Strategic feature creation significantly improves model performance

Imbalance Handling: ADASYN effectively addresses 95:5 class imbalance

Practical Utility: System identifies 23 high-risk transactions for investigation
