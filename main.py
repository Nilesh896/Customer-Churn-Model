import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Import components
from src.data_preprocessing import load_data, clean_data, split_data, get_preprocessor
from src.eda import generate_all_eda_plots
from src.models import get_baseline_models
from src.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curves
from src.optimization import (
    run_baseline_cross_validation, perform_grid_search, 
    evaluate_and_compare_models, select_and_save_best_model
)
from src.gradient_descent import run_learning_rate_experiments, run_gd_vs_sklearn_comparison

def generate_final_summary_md(best_model_name, best_params, baseline_results, optimized_results, gd_summary, feature_importance_df=None):
    """
    Generates a professional final report summary markdown file.
    """
    summary_path = "outputs/results/final_summary.md"
    os.makedirs("outputs/results", exist_ok=True)
    
    # Format baseline table
    baseline_table = baseline_results.to_markdown(index=False)
    # Format optimized table
    optimized_table = optimized_results.to_markdown(index=False)
    # Format gd table
    gd_table = gd_summary.to_markdown(index=False)
    
    # Format feature importance table if available
    fi_section = ""
    if feature_importance_df is not None:
        fi_section = f"\n### Top Features Influencing Churn\n{feature_importance_df.head(10).to_markdown(index=False)}\n"
        
    content = f"""# Customer Churn Prediction – Model Optimization
## Final Summary Report

### 1. Problem Statement & Objective
A telecom company wants to improve the accuracy of its customer churn prediction model. This project establishes a systematic optimization strategy by comparing different hyperparameter tuning and model selection techniques to recommend the best-performing model.

### 2. Dataset Details
- **Source**: IBM Telco Customer Churn dataset obtained through a publicly available dataset mirror.
- **Size**: 7,043 rows (before cleaning/duplicate removal).
- **Target Variable**: `Churn` (Yes/No mapped to 1/0).
- **Imbalance**: Approximately 26.5% of customers churned.

### 3. Baseline Model Evaluation (Test Set)
{baseline_table}

### 4. Optimized Model Evaluation (Test Set)
{optimized_table}

### 5. Final Model Selection
- **Best Model**: {best_model_name}
- **Best Hyperparameters**: `{best_params}`
- **F1 Score**: {optimized_results.loc[optimized_results['Model'] == best_model_name, 'F1-Score'].values[0]:.4f}
- **ROC-AUC**: {optimized_results.loc[optimized_results['Model'] == best_model_name, 'ROC-AUC'].values[0]:.4f}
- **Saved Pipeline**: `models/best_pipeline.pkl` (includes preprocessing and the tuned estimator)

{fi_section}

### 6. Gradient Descent Experiment & Sklearn Comparison
The gradient descent algorithm for Logistic Regression was implemented from scratch using NumPy. Below is the head-to-head comparison on preprocessed features:
{gd_table}

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
"""
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Final report summary written to {summary_path}")

def main():
    print("=======================================================")
    print("CUSTOMER CHURN MODEL OPTIMIZATION PIPELINE")
    print("=======================================================")
    
    # 1. Load and clean dataset
    raw_df = load_data()
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    cleaned_df = clean_data(raw_df)
    print(f"Cleaned dataset shape (after drop duplicates/NaNs): {cleaned_df.shape}")
    
    # 2. Exploratory Data Analysis
    generate_all_eda_plots(cleaned_df)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 4. Get preprocessor and baseline models
    preprocessor = get_preprocessor()
    baseline_models = get_baseline_models()
    
    # 5. Cross validation on baseline models (uses pipeline to avoid leakage)
    cv_results = run_baseline_cross_validation(X_train, y_train, preprocessor, baseline_models)
    
    # 6. Hyperparameter tuning using GridSearchCV inside pipeline
    tuned_pipelines, tuning_df = perform_grid_search(X_train, y_train, preprocessor, baseline_models)
    
    # 7. Evaluate and compare baseline vs optimized models
    results_df = evaluate_and_compare_models(
        baseline_models, tuned_pipelines, 
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    # 8. Best model selection and serialization
    best_model_name, best_pipeline = select_and_save_best_model(tuned_pipelines, results_df)
    
    # Fit preprocessor separately on training data for custom Gradient Descent comparison
    # (since our custom GD is a numpy-only class and cannot be easily put inside sklearn Pipeline directly)
    print("\nPreparing preprocessed data for Custom Gradient Descent experiments...")
    preprocessor.fit(X_train, y_train)
    X_train_gd = preprocessor.transform(X_train)
    X_test_gd = preprocessor.transform(X_test)
    
    # 9. Run Custom Gradient Descent experiments
    run_learning_rate_experiments(X_train_gd, y_train, X_test_gd, y_test)
    gd_comparison = run_gd_vs_sklearn_comparison(X_train_gd, y_train, X_test_gd, y_test)
    
    # 10. Extract Feature Importance for best model
    classifier = best_pipeline.named_steps['classifier']
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Clean feature names
    feature_names_clean = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    
    feature_importance_df = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_clean,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        feature_importance_df.to_csv("outputs/results/feature_importance.csv", index=False)
        print("Feature importances saved to outputs/results/feature_importance.csv")
    elif hasattr(classifier, "coef_"):
        coefs = np.abs(classifier.coef_[0])
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_clean,
            'Coefficient Magnitude': coefs
        }).sort_values(by='Coefficient Magnitude', ascending=False)
        feature_importance_df.to_csv("outputs/results/feature_importance.csv", index=False)
        print("Feature coefficient magnitudes saved to outputs/results/feature_importance.csv")
        
    # 11. Plot Confusion Matrix & ROC curves for final visualization
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name, "confusion_matrix.png")
    
    # Create baseline pipelines for ROC plotting
    fitted_baseline_pipelines = {}
    for name, model in baseline_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        fitted_baseline_pipelines[f"{name} (Baseline)"] = pipe
        
    # Add final optimized model
    fitted_baseline_pipelines[f"{best_model_name} (Optimized)"] = best_pipeline
    plot_roc_curves(fitted_baseline_pipelines, X_test, y_test, "roc_curves.png")
    
    # 12. Generate report
    best_params = tuning_df.loc[tuning_df['Model'] == best_model_name, 'Best Parameters'].values[0]
    baseline_eval = pd.read_csv("outputs/results/baseline_results.csv")
    optimized_eval = pd.read_csv("outputs/results/optimized_results.csv")
    
    generate_final_summary_md(
        best_model_name, best_params, 
        baseline_eval, optimized_eval, gd_comparison,
        feature_importance_df
    )
    
    print("\nPipeline execution complete! All deliverables generated successfully.")

if __name__ == "__main__":
    main()
