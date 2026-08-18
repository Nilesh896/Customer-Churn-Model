import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.evaluation import calculate_metrics

class LogisticRegressionGD:
    """
    Logistic Regression classifier trained from scratch using Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        # Clip values to avoid overflow/underflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
        
    def fit(self, X, y):
        m, n = X.shape
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []
        
        y = np.array(y)
        
        for i in range(self.iterations):
            # Forward pass: calculate predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Calculate binary cross entropy loss
            # Add small epsilon to prevent log(0)
            eps = 1e-15
            loss = - (1 / m) * np.sum(y * np.log(y_predicted + eps) + (1 - y) * np.log(1 - y_predicted + eps))
            self.loss_history.append(loss)
            
            # Calculate gradients
            dw = (1 / m) * np.dot(X.T, (y_predicted - y))
            db = (1 / m) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self
        
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return np.where(probs >= threshold, 1, 0)

def run_learning_rate_experiments(X_train, y_train, X_test, y_test):
    """
    Runs Gradient Descent training across different learning rates: 0.001, 0.01, 0.05, 0.1.
    Records actual final loss, accuracy, precision, recall, F1, and creates a loss-vs-iterations plot.
    """
    print("\nRunning custom Gradient Descent learning rate experiments...")
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    iterations = 1000
    
    lr_records = []
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for lr in learning_rates:
        model = LogisticRegressionGD(learning_rate=lr, iterations=iterations)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        # Calculate test metrics
        metrics = calculate_metrics(y_test, y_pred, probs)
        
        lr_records.append({
            'Learning Rate': lr,
            'Final Loss': model.loss_history[-1],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score'],
            'Iterations': iterations
        })
        
        # Plot loss history
        ax.plot(range(1, iterations + 1), model.loss_history, label=f'LR = {lr}')
        
    ax.set_title("Gradient Descent Loss Convergence by Learning Rate")
    ax.set_xlabel("Iterations (Epochs)")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_yscale('log') # Use log scale if helpful for visual range
    ax.legend()
    
    # Save chart
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "gd_learning_rate_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gradient Descent learning rate plot saved to {fig_path}")
    
    lr_df = pd.DataFrame(lr_records)
    lr_df.to_csv("outputs/results/gd_learning_rates.csv", index=False)
    print("Gradient Descent learning rate details saved to outputs/results/gd_learning_rates.csv")
    
    return lr_df

def run_gd_vs_sklearn_comparison(X_train, y_train, X_test, y_test):
    """
    Compares our custom Gradient Descent Logistic Regression (at LR=0.1)
    against Scikit-learn's Logistic Regression model.
    """
    print("\nRunning Head-to-Head Comparison: Custom GD vs Scikit-learn Logistic Regression...")
    
    # 1. Custom GD (using best learning rate 0.1)
    custom_model = LogisticRegressionGD(learning_rate=0.1, iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_pred = custom_model.predict(X_test)
    custom_prob = custom_model.predict_proba(X_test)
    custom_metrics = calculate_metrics(y_test, custom_pred, custom_prob)
    custom_metrics['Model'] = 'Custom GD (LR=0.1)'
    
    # 2. Sklearn Logistic Regression
    sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_prob = sklearn_model.predict_proba(X_test)[:, 1]
    sklearn_metrics = calculate_metrics(y_test, sklearn_pred, sklearn_prob)
    sklearn_metrics['Model'] = 'Scikit-learn LogisticRegression'
    
    # Create comparison table
    comparison_df = pd.DataFrame([sklearn_metrics, custom_metrics])
    
    # Reorder columns
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    comparison_df = comparison_df[cols]
    
    comparison_df.to_csv("outputs/results/gd_experiment_results.csv", index=False)
    print("Head-to-Head Comparison saved to outputs/results/gd_experiment_results.csv")
    print(comparison_df)
    
    return comparison_df
