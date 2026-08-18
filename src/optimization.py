import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from src.evaluation import calculate_metrics, plot_confusion_matrix

def run_baseline_cross_validation(X_train, y_train, preprocessor, models):
    """
    Performs Stratified 5-Fold Cross Validation on baseline models.
    To prevent data leakage, the preprocessor and estimator are combined into an sklearn Pipeline.
    """
    print("Running Stratified 5-Fold Cross Validation on baseline models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    
    for model_name, model in models.items():
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Cross validate
        scores = cross_validate(
            pipeline, X_train, y_train, 
            cv=cv, 
            scoring=['accuracy', 'f1', 'roc_auc'],
            n_jobs=-1
        )
        
        cv_records.append({
            'Model': model_name,
            'CV Accuracy Mean': np.mean(scores['test_accuracy']),
            'CV Accuracy Std': np.std(scores['test_accuracy']),
            'CV F1 Mean': np.mean(scores['test_f1']),
            'CV ROC-AUC Mean': np.mean(scores['test_roc_auc'])
        })
        
    cv_df = pd.DataFrame(cv_records)
    print("Cross Validation Results:")
    print(cv_df)
    return cv_df

def get_hyperparameter_grids():
    """
    Returns parameter grids for GridSearchCV.
    Grid keys match parameter names inside the Pipeline ('classifier__<param_name>').
    """
    grids = {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2']
        },
        'Decision Tree': {
            'classifier__max_depth': [3, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10],
            'classifier__min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5]
        }
    }
    return grids

def perform_grid_search(X_train, y_train, preprocessor, baseline_models):
    """
    Performs hyperparameter tuning for each classifier using GridSearchCV.
    GridSearchCV is fed with a Pipeline containing the preprocessor and the classifier.
    Scoring is optimized based on the F1-Score.
    """
    print("\nStarting Hyperparameter Tuning with Grid Search...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grids = get_hyperparameter_grids()
    tuned_pipelines = {}
    tuning_details = []
    
    for model_name, model in baseline_models.items():
        if model_name not in grids:
            print(f"Skipping parameter tuning for {model_name} (no grid specified).")
            continue
            
        print(f"Tuning {model_name}...")
        
        # Create nested pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Grid Search CV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grids[model_name],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        
        tuned_pipelines[model_name] = grid_search.best_estimator_
        tuning_details.append({
            'Model': model_name,
            'Best Parameters': str(grid_search.best_params_),
            'Best CV F1': grid_search.best_score_
        })
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
    tuning_df = pd.DataFrame(tuning_details)
    return tuned_pipelines, tuning_df

def evaluate_and_compare_models(baseline_models, tuned_pipelines, X_train, y_train, X_test, y_test, preprocessor):
    """
    Evaluates both baseline and optimized models on the test dataset.
    Generates comparison dataframes and plots.
    """
    print("\nEvaluating baseline and optimized models on the test set...")
    
    results = []
    
    # 1. Evaluate Baselines
    for name, model in baseline_models.items():
        # Fit baseline pipeline on training data
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = name
        metrics['Stage'] = 'Baseline'
        results.append(metrics)
        
    # 2. Evaluate Optimized Models
    for name, pipeline in tuned_pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = name
        metrics['Stage'] = 'Optimized'
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    
    # Split baseline and optimized into separate files
    baseline_results = results_df[results_df['Stage'] == 'Baseline'].drop(columns=['Stage'])
    optimized_results = results_df[results_df['Stage'] == 'Optimized'].drop(columns=['Stage'])
    
    # Save outputs
    os.makedirs("outputs/results", exist_ok=True)
    baseline_results.to_csv("outputs/results/baseline_results.csv", index=False)
    optimized_results.to_csv("outputs/results/optimized_results.csv", index=False)
    print("Baseline test results saved to outputs/results/baseline_results.csv")
    print("Optimized test results saved to outputs/results/optimized_results.csv")
    
    # Plot comparisons
    plot_optimization_comparison(results_df)
    
    return results_df

def plot_optimization_comparison(results_df):
    """
    Creates bar charts comparing performance metrics before and after optimization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ['Accuracy', 'F1-Score', 'ROC-AUC']
    
    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(
            data=results_df, 
            x='Model', 
            y=metric, 
            hue='Stage', 
            ax=axes[i],
            palette='Blues_d'
        )
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel('Model')
        axes[i].set_ylim(0, 1.05)
        # Add labels on bars
        for p in axes[i].patches:
            height = p.get_height()
            if height > 0:
                axes[i].annotate(f'{height:.2f}',
                            xy=(p.get_x() + p.get_width() / 2, height + 0.01),
                            ha='center', va='bottom', fontsize=9)
                            
    plt.tight_layout()
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "baseline_vs_optimized_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Optimization comparison plot saved to {filepath}")

def select_and_save_best_model(tuned_pipelines, results_df):
    """
    Selects the best performing pipeline on the test set.
    Primary criteria: F1-Score.
    Secondary criteria: ROC-AUC.
    Saves the complete winning pipeline to models/best_pipeline.pkl.
    """
    opt_df = results_df[results_df['Stage'] == 'Optimized'].copy()
    
    # Sort by F1-Score descending, and ROC-AUC descending
    opt_sorted = opt_df.sort_values(by=['F1-Score', 'ROC-AUC'], ascending=False)
    best_row = opt_sorted.iloc[0]
    best_model_name = best_row['Model']
    best_f1 = best_row['F1-Score']
    best_auc = best_row['ROC-AUC']
    
    best_pipeline = tuned_pipelines[best_model_name]
    
    os.makedirs("models", exist_ok=True)
    pipeline_path = "models/best_pipeline.pkl"
    joblib.dump(best_pipeline, pipeline_path)
    
    print(f"\n=======================================================")
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"Test F1-Score: {best_f1:.4f}")
    print(f"Test ROC-AUC: {best_auc:.4f}")
    print(f"Saved complete pipeline to {pipeline_path}")
    print(f"=======================================================")
    
    return best_model_name, best_pipeline
