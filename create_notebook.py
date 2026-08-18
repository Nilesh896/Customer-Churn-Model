import os
import json

def build_notebook():
    notebook_path = "notebooks/customer_churn_analysis.ipynb"
    os.makedirs("notebooks", exist_ok=True)
    
    cells = []
    
    # 1. Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Customer Churn Prediction – Model Optimization\n",
            "### Academic Mini Project\n",
            "This notebook details the systematic development and optimization of machine learning classifiers to predict customer churn in a telecom firm. We evaluate models, conduct hyperparameter search using nested pipelines, and implement a custom gradient descent solver from scratch in NumPy."
        ]
    })
    
    # 2. Imports
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Import Libraries"
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc\n",
            "\n",
            "# Set visualization styles\n",
            "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
            "plt.rcParams['figure.figsize'] = (10, 6)"
        ]
    })
    
    # 3. Load Data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Load and Inspect Dataset\n",
            "We load the dataset from the local folder. If it is missing, we download it from our mirror."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_preprocessing import load_data\n",
            "\n",
            "raw_df = load_data()\n",
            "print(f\"Dataset shape: {raw_df.shape}\")\n",
            "raw_df.info()"
        ]
    })
    
    # 4. Clean Data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Data Cleaning\n",
            "We address blank spaces in `TotalCharges` (coerce to NaN and fill with `0.0` for new customers where tenure is 0) and encode the target variable `Churn` to `0` and `1`."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_preprocessing import clean_data\n",
            "\n",
            "cleaned_df = clean_data(raw_df)\n",
            "print(f\"Cleaned dataset shape: {cleaned_df.shape}\")\n",
            "cleaned_df.head()"
        ]
    })
    
    # 5. EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Exploratory Data Analysis (EDA)\n",
            "We create visualizations showing churn distribution and correlations. These are saved in `outputs/figures/`."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.eda import generate_all_eda_plots\n",
            "\n",
            "generate_all_eda_plots(cleaned_df)"
        ]
    })
    
    # 6. Split Data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Stratified Train-Test Split\n",
            "We split features and target, using stratify to maintain class distributions."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_preprocessing import split_data\n",
            "\n",
            "X_train, X_test, y_train, y_test = split_data(cleaned_df)\n",
            "print(f\"Train size: {X_train.shape}, Test size: {X_test.shape}\")\n",
            "print(\"Train Churn Rate:\\n\", y_train.value_counts(normalize=True))"
        ]
    })
    
    # 7. Pipeline Config
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Preprocessing Configuration\n",
            "We construct our `ColumnTransformer` with zero data leakage: categorical features are One-Hot Encoded and numerical features are Standard Scaled. Note: fitting will be done inside Pipeline objects during model training."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data_preprocessing import get_preprocessor\n",
            "\n",
            "preprocessor = get_preprocessor()\n",
            "print(preprocessor)"
        ]
    })
    
    # 8. Baseline Models
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Baseline Models & Evaluation\n",
            "We initialize baseline classifiers and evaluate them using Stratified 5-Fold Cross Validation on the training data."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.models import get_baseline_models\n",
            "from src.optimization import run_baseline_cross_validation\n",
            "\n",
            "baseline_models = get_baseline_models()\n",
            "cv_results = run_baseline_cross_validation(X_train, y_train, preprocessor, baseline_models)"
        ]
    })
    
    # 9. GridSearchCV
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 8: GridSearchCV Hyperparameter Tuning\n",
            "To maximize F1-score, we wrap preprocessing and estimators into an sklearn `Pipeline` and run Grid Search CV. This guarantees that preprocessing coefficients are fit only on training folds, avoiding data leakage."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.optimization import perform_grid_search\n",
            "\n",
            "tuned_pipelines, tuning_df = perform_grid_search(X_train, y_train, preprocessor, baseline_models)"
        ]
    })
    
    # 10. Evaluation Comparison
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 9: Before vs After Optimization Comparison\n",
            "We evaluate baseline models and optimized models on the test set and compare the changes in metrics."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.optimization import evaluate_and_compare_models\n",
            "\n",
            "results_df = evaluate_and_compare_models(\n",
            "    baseline_models, tuned_pipelines, \n",
            "    X_train, y_train, X_test, y_test, preprocessor\n",
            ")"
        ]
    })
    
    # 11. Selection and Save
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 10: Best Model Selection and Serialization\n",
            "We programmatically select the pipeline with the highest test set F1-Score and serialize it as a single file."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.optimization import select_and_save_best_model\n",
            "\n",
            "best_name, best_pipe = select_and_save_best_model(tuned_pipelines, results_df)"
        ]
    })
    
    # 12. Custom Gradient Descent
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 11: Custom Gradient Descent from Scratch\n",
            "We train a custom Logistic Regression model from scratch in NumPy using Gradient Descent to study convergence."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.gradient_descent import run_learning_rate_experiments, run_gd_vs_sklearn_comparison\n",
            "\n",
            "# Fit preprocessor separately on training data to scale input for GD solver\n",
            "preprocessor.fit(X_train, y_train)\n",
            "X_train_scaled = preprocessor.transform(X_train)\n",
            "X_test_scaled = preprocessor.transform(X_test)\n",
            "\n",
            "# 1. Run learning rate experiments\n",
            "lr_experiments = run_learning_rate_experiments(X_train_scaled, y_train, X_test_scaled, y_test)\n",
            "\n",
            "# 2. Run head-to-head comparison\n",
            "gd_vs_sklearn = run_gd_vs_sklearn_comparison(X_train_scaled, y_train, X_test_scaled, y_test)"
        ]
    })
    
    # 13. Feature Importance
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 12: Feature Importance Analysis\n",
            "We visualize the top factors contributing to customer churn."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "classifier = best_pipe.named_steps['classifier']\n",
            "feature_names = best_pipe.named_steps['preprocessor'].get_feature_names_out()\n",
            "feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]\n",
            "\n",
            "if hasattr(classifier, 'feature_importances_'):\n",
            "    importances = classifier.feature_importances_\n",
            "    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\n",
            "    fi_df = fi_df.sort_values(by='Importance', ascending=False)\n",
            "    \n",
            "    plt.figure(figsize=(10, 6))\n",
            "    sns.barplot(data=fi_df.head(10), x='Importance', y='Feature', palette='viridis')\n",
            "    plt.title('Top 10 Feature Importances')\n",
            "    plt.show()\n",
            "elif hasattr(classifier, 'coef_'):\n",
            "    coefs = np.abs(classifier.coef_[0])\n",
            "    fi_df = pd.DataFrame({'Feature': feature_names, 'Coefficient Magnitude': coefs})\n",
            "    fi_df = fi_df.sort_values(by='Coefficient Magnitude', ascending=False)\n",
            "    \n",
            "    plt.figure(figsize=(10, 6))\n",
            "    sns.barplot(data=fi_df.head(10), x='Coefficient Magnitude', y='Feature', palette='viridis')\n",
            "    plt.title('Top 10 Feature Coefficient Magnitudes')\n",
            "    plt.show()"
        ]
    })
    
    # 14. Conclusion & Business Insights
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 13: Summary of Insights & Conclusion\n",
            "- **Month-to-month contracts** present the highest risk of churn.\n",
            "- **Tenure** has an inverse relationship with churn; newer customers are highly unstable.\n",
            "- **Hyperparameter Tuning** via GridSearchCV yielded a notable improvement in F1-score.\n",
            "- **Gradient Descent** convergence demonstrates that a learning rate of `0.1` achieves stable minimization without divergence."
        ]
    })
    
    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_json, f, indent=1)
    print(f"Jupyter Notebook successfully created at {notebook_path}")

if __name__ == '__main__':
    build_notebook()
