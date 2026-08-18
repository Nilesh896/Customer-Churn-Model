# Viva Preparation Guide: Customer Churn Prediction – Model Optimization

This document contains 25 common viva questions and detailed, concise answers tailored for a college-level machine learning mini-project defense.

---

### Q1. What is customer churn, and why is it important for a business?
**Answer:** Customer churn refers to the phenomenon where customers stop doing business with an entity (e.g., cancelling a subscription or ending service). Predictively identifying churn allows telecom companies to proactively target at-risk customers with retention offers, which is much cheaper than acquiring new customers.

### Q2. Why is customer churn prediction formulated as a classification problem?
**Answer:** Churn prediction is a classification problem because the target variable is categorical and discrete—specifically binary. The goal is to classify each customer into one of two groups: **Churned (1/Yes)** or **Retained (0/No)**.

### Q3. What is the role of Logistic Regression in classification, and why did you use it?
**Answer:** Logistic Regression is a statistical model that predicts the probability of a binary event using the logistic (sigmoid) function. It serves as a strong, interpretable baseline model. Its coefficients explain the impact and direction (positive/negative association) of features on the log-odds of churn.

### Q4. How does a Decision Tree Classifier work?
**Answer:** A Decision Tree splits the dataset into subsets based on feature values that maximize information gain (using measures like Gini impurity or Entropy). It builds a tree-like model of decisions, which is highly readable but prone to overfitting if not regularized (e.g., by setting `max_depth`).

### Q5. What is a Random Forest Classifier, and how does it reduce overfitting compared to a single Decision Tree?
**Answer:** Random Forest is an ensemble learning method that builds multiple decision trees on random subsets of the data (bagging) and features. It aggregates their predictions (voting/average), which reduces variance and overfitting because individual trees fail in different ways, canceling out random errors.

### Q6. What is a hyperparameter, and how does it differ from a model parameter?
**Answer:**
- **Model Parameters** are learned automatically from the training data during model fitting (e.g., weights in logistic regression or split thresholds in decision trees).
- **Hyperparameters** are configurations set by the engineer before training begins to guide the learning process (e.g., regularization strength `C` in Logistic Regression, or `n_estimators` in Random Forest).

### Q7. What is hyperparameter tuning, and why is it necessary?
**Answer:** Hyperparameter tuning is the process of finding the optimal set of hyperparameters that yields the best model performance on unseen data. It is necessary because the default hyperparameters are generic and rarely optimal for a specific custom dataset.

### Q8. What is Grid Search (GridSearchCV), and how does it work?
**Answer:** Grid Search is a systematic optimization technique that searches through a manually specified grid of hyperparameter combinations. It trains a model for every possible combination, evaluates performance using cross-validation, and returns the combination that scored highest.

### Q9. What is Cross Validation, and why is it preferable to a single train-test split?
**Answer:** Cross Validation partitions the training data into multiple folds (e.g., 5 folds). The model is trained on 4 folds and validated on the remaining fold, repeating the process 5 times. It is preferable because it uses all data for both training and validation, providing a more stable and generalized estimate of model performance with a standard deviation.

### Q10. Why is Stratified K-Fold Cross Validation specifically chosen for churn prediction?
**Answer:** Churn datasets are typically imbalanced (fewer churned customers than retained ones). Stratified K-Fold ensures that the original proportion of the target classes is preserved in each fold. This prevents folds from having too few or no churn instances, which would lead to biased and unreliable metric estimates.

### Q11. What is Gradient Descent, and how does it work?
**Answer:** Gradient Descent is an iterative optimization algorithm used to minimize a loss function (like Binary Cross-Entropy). It calculates the gradient (derivatives) of the loss function with respect to the weights, and updates the weights in the opposite direction of the gradient to find the minimum.

### Q12. What is the learning rate in Gradient Descent?
**Answer:** The learning rate ($\alpha$) is a step-size hyperparameter that controls how much the weights are adjusted in each iteration. It dictates how fast or slow the model climbs down the loss slope toward the global minimum.

### Q13. What happens if the learning rate is too high?
**Answer:** If the learning rate is too high, the updates to the weights will be too large. The algorithm may overshoot the minimum, causing the loss to oscillate or even diverge (increase) instead of decreasing.

### Q14. What happens if the learning rate is too low?
**Answer:** If the learning rate is too low, weight updates will be extremely small. The model will require many iterations (epochs) to reach the minimum, leading to high training times and potential stagnation in local minima.

### Q15. Why is Accuracy not a sufficient metric for evaluating churn models?
**Answer:** Telecom datasets are imbalanced. If only 10% of customers churn, a naive model that predicts "No Churn" for everyone would achieve 90% accuracy but would be completely useless at detecting churners. Therefore, metrics like F1-Score, Precision, and Recall are necessary.

### Q16. What is the difference between Precision and Recall, and what is F1-Score?
**Answer:**
- **Precision**: Of all customers predicted to churn, how many actually did? (Minimizes False Positives).
- **Recall**: Of all customers who actually churned, how many did the model detect? (Minimizes False Negatives).
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single balanced metric for imbalanced classes.

### Q17. What is ROC-AUC, and what does it measure?
**Answer:** ROC-AUC stands for Receiver Operating Characteristic – Area Under Curve. The ROC curve plots the True Positive Rate vs the False Positive Rate at various classification thresholds. The AUC (Area Under Curve) measures the model's ability to distinguish between classes. An AUC of 1.0 is perfect; 0.5 is equivalent to random guessing.

### Q18. What is overfitting, and how did you detect/prevent it?
**Answer:** Overfitting occurs when a model learns noise and details in the training set too well, leading to poor generalization on unseen data. We detected it by comparing training CV scores with test set scores. We prevented it using K-Fold cross-validation, hyperparameter grid search (such as restricting `max_depth` in Decision Trees), and ensemble methods like Random Forests.

### Q19. What is data leakage, and how does your project prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) is used to train the model. We prevented it by:
1. Performing the train-test split *before* any feature processing.
2. Embedding preprocessing (`StandardScaler`, `OneHotEncoder`) and estimators together inside an sklearn `Pipeline`.
3. Passing this complete pipeline to `GridSearchCV` so that scaling/encoding parameters are fit *only* on the training folds of each cross-validation loop.

### Q20. Why do we scale numerical features like MonthlyCharges and tenure?
**Answer:** We scale numerical features so that they are on a comparable scale (typically mean=0, std=1). Algorithms that rely on distances or gradient calculations (like Logistic Regression and Custom Gradient Descent) converge faster and prevent features with larger scales (like `TotalCharges` in thousands) from dominating the loss updates.

### Q21. Why is a False Negative particularly costly for a telecom churn scenario?
**Answer:** A False Negative occurs when a customer who is about to churn is predicted by the model to stay. The business takes no action, and the customer leaves. The loss of customer revenue is typically far greater than the cost of a promotional retention offer (which is the cost of a False Positive).

### Q22. How did you select the final model for deployment?
**Answer:** We systematically evaluated all baseline models, optimized them using GridSearchCV, and compared them on the test dataset. The final model was chosen automatically by selecting the pipeline that yielded the highest **F1-Score** (primary metric) on the test set, using ROC-AUC as a secondary tie-breaker.

### Q23. Why did you package the preprocessor and the classifier into a single pipeline file?
**Answer:** Saving the preprocessor and estimator together as a single `models/best_pipeline.pkl` guarantees that raw inputs (e.g. from the Streamlit web form) go through the exact same transformations as the training set, eliminating transformation mismatches and coding errors during deployment.

### Q24. What are some of the key features influencing churn in your model?
**Answer:** Based on feature importances (or coefficients), the key features include:
- **Contract**: Month-to-month contracts are highly associated with churn.
- **Tenure**: Low tenure strongly correlates with a high likelihood of churn.
- **Internet Service**: Customers with Fiber Optic service tend to have a higher observed churn.
- **Payment Method**: Electronic Check payments are associated with higher churn rates.

### Q25. What is the risk level thresholding logic used in your prediction tab?
**Answer:** We classify predicted churn probability into three actionable risk categories:
- **Low Risk** (< 30% probability): Standard communication.
- **Medium Risk** (30% to 69.99% probability): Mild proactive engagement (customer newsletters/feedback).
- **High Risk** ($\ge$ 70% probability): Critical target list for immediate retention campaigns (discounts, personal calls).
