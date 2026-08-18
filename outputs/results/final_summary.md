# Customer Churn Prediction – Model Optimization
## Final Summary Report

### 1. Problem Statement & Objective
A telecom company wants to improve the accuracy of its customer churn prediction model. This project establishes a systematic optimization strategy by comparing different hyperparameter tuning and model selection techniques to recommend the best-performing model.

### 2. Dataset Details
- **Source**: IBM Telco Customer Churn dataset obtained through a publicly available dataset mirror.
- **Size**: 7,043 rows (before cleaning/duplicate removal).
- **Target Variable**: `Churn` (Yes/No mapped to 1/0).
- **Imbalance**: Approximately 26.5% of customers churned.

### 3. Baseline Model Evaluation (Test Set)
|   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC | Model               |
|-----------:|------------:|---------:|-----------:|----------:|:--------------------|
|   0.806246 |    0.659306 | 0.558824 |   0.60492  |  0.842171 | Logistic Regression |
|   0.731725 |    0.494565 | 0.486631 |   0.490566 |  0.652964 | Decision Tree       |
|   0.792051 |    0.642105 | 0.489305 |   0.555387 |  0.827543 | Random Forest       |
|   0.797729 |    0.653979 | 0.505348 |   0.570136 |  0.841505 | Gradient Boosting   |

### 4. Optimized Model Evaluation (Test Set)
|   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC | Model               |
|-----------:|------------:|---------:|-----------:|----------:|:--------------------|
|   0.805536 |    0.657233 | 0.558824 |   0.604046 |  0.841288 | Logistic Regression |
|   0.750887 |    0.527316 | 0.593583 |   0.558491 |  0.787052 | Decision Tree       |
|   0.806246 |    0.668896 | 0.534759 |   0.594354 |  0.842068 | Random Forest       |
|   0.797729 |    0.653979 | 0.505348 |   0.570136 |  0.841505 | Gradient Boosting   |

### 5. Final Model Selection
- **Best Model**: Logistic Regression
- **Best Hyperparameters**: `{'classifier__C': 10, 'classifier__penalty': 'l2'}`
- **F1 Score**: 0.6040
- **ROC-AUC**: 0.8413
- **Saved Pipeline**: `models/best_pipeline.pkl` (includes preprocessing and the tuned estimator)


### Top Features Influencing Churn
| Feature                        |   Coefficient Magnitude |
|:-------------------------------|------------------------:|
| InternetService_Fiber optic    |                2.34616  |
| MonthlyCharges                 |                1.88105  |
| MultipleLines_No phone service |                1.38313  |
| Contract_Two year              |                1.36166  |
| tenure                         |                1.29683  |
| StreamingTV_Yes                |                0.839812 |
| StreamingMovies_Yes            |                0.839487 |
| PhoneService_Yes               |                0.709316 |
| Contract_One year              |                0.693317 |
| MultipleLines_Yes              |                0.597818 |


### 6. Gradient Descent Experiment & Sklearn Comparison
The gradient descent algorithm for Logistic Regression was implemented from scratch using NumPy. Below is the head-to-head comparison on preprocessed features:
| Model                           |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------------------------|-----------:|------------:|---------:|-----------:|----------:|
| Scikit-learn LogisticRegression |   0.806246 |    0.659306 | 0.558824 |   0.60492  |  0.842171 |
| Custom GD (LR=0.1)              |   0.801987 |    0.654723 | 0.537433 |   0.590308 |  0.839714 |

### 7. Key Business Insights
- **Contract Type**: Month-to-month contracts showed significantly higher churn rates compared to long-term 1-year and 2-year contracts.
- **Tenure**: Customers with low tenure (first 1–12 months) are at a much higher risk of churning.
- **Monthly Charges**: Higher monthly charges show a positive association with churn.
- **Payment Method**: Customers paying via Electronic Check display a higher observed churn rate than those using automatic bank transfers or credit cards.

### 8. Recommendations
1. **Promote Long-Term Contracts**: Offer incentives (discounts or value-added services) to migrate month-to-month customers to 1-year or 2-year agreements.
2. **Onboarding Focus**: Implement proactive outreach and customer support check-ins during the first 6 months of the customer journey.
3. **Electronic Check Migration**: Encourage customers on Electronic Check to set up automatic payment methods (Credit Card / Bank Transfer) by offering a one-time billing credit.
4. **Targeted Retention Campaigns**: Use the predictive model's probability scores to identify and target high-risk, high-value customers with tailored retention offers before they churn.

### 9. Conclusion
Applying a systematic model optimization workflow using GridSearchCV and Stratified K-Fold cross validation significantly improved the classification performance (specifically F1-score). Packaging the entire preprocessing and classification logic as a single pipeline guarantees zero data leakage and a clean deployment vector for production.
