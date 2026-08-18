# Customer Churn Prediction – Model Optimization

An end-to-end, college-level Machine Learning mini-project demonstrating a systematic optimization strategy to predict telecom customer churn. The project highlights the performance improvements achieved by applying Cross Validation, Hyperparameter Tuning (Grid Search), and comparing baseline classifiers against optimized pipelines. Additionally, it implements a custom Gradient Descent Logistic Regression solver from scratch in NumPy for educational analysis.

---

## 📖 Problem Statement

> A telecom company wants to improve the accuracy of its customer churn prediction model. Develop a systematic optimization strategy by comparing different hyperparameter tuning and model selection techniques to recommend the best-performing model.

---

## 🎯 Project Objective

1. **Systematic Preprocessing:** Develop a pipeline mapping raw customer attributes to standard features with zero data leakage.
2. **Baseline Modeling:** Train baseline models (Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting).
3. **Robust Evaluation:** Compare models on multiple classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. **Cross-Validation:** Implement Stratified 5-Fold Cross Validation to ensure statistical stability.
5. **Hyperparameter Tuning:** Conduct grid searches on classifier pipelines using `GridSearchCV` optimized for F1-Score.
6. **Gradient Descent Analysis:** Implement a numpy-based Logistic Regression solver from scratch to examine how learning rates control convergence and compare results to scikit-learn.
7. **Interactive Deployment:** Package the winning classifier and preprocessor into a single serialized file and expose it through an interactive Streamlit dashboard.

---

## 🛠️ Technologies Used

* **Python 3.10+** (Core programming language)
* **Pandas** (Data manipulation and cleaning)
* **NumPy** (Linear algebra & custom Gradient Descent implementation)
* **Scikit-learn** (Pipelines, encoders, scalers, models, metrics, and tuning)
* **Matplotlib & Seaborn** (Data visualization and figure saving)
* **Streamlit** (Interactive user dashboard)
* **Joblib** (Model serialization)
* **Plotly** (Interactive indicators and gauges)

---

## 📊 Dataset

* **Source**: IBM Telco Customer Churn dataset obtained through a publicly available dataset mirror.
* **Size**: 7,043 rows, 21 columns.
* **Target Variable**: `Churn` (encoded as `1` for Yes, `0` for No).
* **Imbalance**: Approx. 26.5% churn rate, necessitating F1-Score as the primary evaluation metric.
* **TotalCharges Handling**: Missing charges (blank spaces) are coerced to `NaN` and imputed with `0.0` (which matches new users with `tenure == 0`).

---

## 📂 Project Structure

```
customer-churn-model-optimization/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── models.py
│   ├── evaluation.py
│   ├── optimization.py
│   ├── gradient_descent.py
│   └── prediction.py
│
├── models/
│   └── best_pipeline.pkl
│
├── outputs/
│   ├── figures/
│   │   ├── churn_distribution.png
│   │   ├── churn_by_contract.png
│   │   ├── churn_by_tenure.png
│   │   ├── churn_by_charges.png
│   │   ├── correlation_matrix.png
│   │   ├── gd_learning_rate_comparison.png
│   │   ├── baseline_vs_optimized_comparison.png
│   │   ├── roc_curves.png
│   │   └── confusion_matrix.png
│   ├── metrics/
│   └── results/
│       ├── baseline_results.csv
│       ├── optimized_results.csv
│       ├── gd_experiment_results.csv
│       └── final_summary.md
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
├── .gitignore
├── VIVA_QUESTIONS.md
└── main.py
```

---

## 🚀 How to Run the Project

Follow these steps to run the pipeline, generate metrics/plots, and launch the interactive dashboard.

### 1. Install Dependencies
Run the package installer using the Python Launcher (`py`):
```powershell
py -m pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the master script to download the dataset, execute data processing, train baseline and optimized models, run Gradient Descent experiments, generate visualizations, and save the best pipeline:
```powershell
py main.py
```

### 3. Launch the Streamlit App
Start the local development server to open the interactive dashboard:
```powershell
py -m streamlit run app/streamlit_app.py
```

---

## ⚙️ Model Optimization Methodology

### Zero Data Leakage Design
To prevent leakage of test or validation data statistics into the model training, we utilize scikit-learn's `Pipeline` API. The `ColumnTransformer` is bundled with the model estimator inside a single pipeline. During cross-validation and Grid Search, the preprocessing parameters (e.g. mean, standard deviation for scaling, and categories for one-hot encoding) are calculated **only** on the training folds.

### GridSearchCV Setup
We search through hyperparameters for Logistic Regression, Decision Trees, and Random Forests. The grid search evaluates configurations using Stratified 5-Fold Cross Validation.

### Custom Gradient Descent Solver
We implement a `LogisticRegressionGD` class from scratch using NumPy. Weights and bias are updated iteratively:
$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
We track BCE loss convergence across learning rates $0.001$, $0.01$, $0.05$, and $0.1$. The optimal custom model is then compared head-to-head with scikit-learn's coordinate-descent solvers.

---

## 📈 Results and Deliverables
After executing `py main.py`, the following files will contain actual results:
* **`outputs/results/baseline_results.csv`**: Metrics of baseline models on the test set.
* **`outputs/results/optimized_results.csv`**: Metrics of tuned pipelines on the test set.
* **`outputs/results/gd_experiment_results.csv`**: Head-to-head comparison of custom GD vs Scikit-learn Logistic Regression.
* **`outputs/results/final_summary.md`**: Academic-style report summary containing best parameters, feature importances, business insights, and recommendations.
* **`outputs/figures/`**: Contains plots of class distributions, correlation matrices, ROC curves, confusion matrices, and learning rate curves.
