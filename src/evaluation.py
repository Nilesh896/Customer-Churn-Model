import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['ROC-AUC'] = np.nan
        
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, filename="confusion_matrix.png"):
    """
    Plots and saves a professional confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (No)', 'Churned (Yes)'],
                yticklabels=['Retained (No)', 'Churned (Yes)'], ax=ax)
                
    # Labels, title and ticks
    ax.set_ylabel('Actual Status', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Status', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, pad=15)
    
    # Text notes to explain what quadrants represent
    # tp, fp, fn, tn definitions
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, 0.2, f"TN: {tn}\n(True Negative)", ha='center', va='center', color='black', fontsize=9)
    ax.text(1.5, 0.2, f"FP: {fp}\n(False Positive)", ha='center', va='center', color='white' if fp > (tn+tp)/4 else 'black', fontsize=9)
    ax.text(0.5, 1.2, f"FN: {fn}\n(False Negative)", ha='center', va='center', color='white' if fn > (tn+tp)/4 else 'black', fontsize=9)
    ax.text(1.5, 1.2, f"TP: {tp}\n(True Positive)", ha='center', va='center', color='white' if tp > (tn+tp)/4 else 'black', fontsize=9)

    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion Matrix saved to {filepath}")
    return cm

def plot_roc_curves(fitted_pipelines, X_test, y_test, filename="roc_curves.png"):
    """
    Plots the ROC curves of all fitted pipelines on the same plot.
    'fitted_pipelines' is a dictionary: {model_name: fitted_pipeline_object}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, pipeline in fitted_pipelines.items():
        # Get probabilities for Churn (class 1)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            # For models that might not have predict_proba (e.g. custom ones without it)
            pass
            
    # Plot random guess line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, pad=15)
    ax.legend(loc="lower right")
    
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC Curves saved to {filepath}")
