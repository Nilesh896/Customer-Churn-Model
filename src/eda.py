import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set cohesive modern style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Primary colors
CHURN_PALETTE = {1: "#E74C3C", 0: "#3498DB"} # Red for Churn, Blue for Stay
CHURN_LABELS = {1: "Churned (Yes)", 0: "Retained (No)"}

def save_plot(fig, filename):
    """
    Helper function to save figure to outputs/figures/
    """
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {filepath}")

def plot_churn_distribution(df):
    """
    Plots the overall churn distribution (Yes vs No).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    # Plot bar plot
    sns.barplot(x=churn_counts.index.map(CHURN_LABELS), y=churn_counts.values, ax=ax, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    
    # Annotate percentage labels
    for i, count in enumerate(churn_counts.values):
        pct = churn_pct.values[i]
        ax.annotate(f"{count} ({pct:.1f}%)", 
                    xy=(i, count + 100), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    ax.set_title("Distribution of Customer Churn")
    ax.set_ylabel("Number of Customers")
    ax.set_xlabel("Customer Status")
    ax.set_ylim(0, max(churn_counts.values) * 1.15)
    
    save_plot(fig, "churn_distribution.png")

def plot_churn_by_contract(df):
    """
    Plots churn distribution split by Contract type.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare cross-tabulation
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn = contract_churn.rename(columns=CHURN_LABELS)
    
    # Plot stacked bar chart
    contract_churn.plot(kind='bar', stacked=True, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], ax=ax, width=0.6)
    
    # Annotate details
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 5: # Only show annotations for large enough sections
            ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', color='white', fontweight='bold')
            
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Contract Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Customer Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_plot(fig, "churn_by_contract.png")

def plot_churn_by_tenure(df):
    """
    Plots the density estimation of customer tenure split by churn.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    sns.kdeplot(data=df[df['Churn'] == 0], x='tenure', label=CHURN_LABELS[0], color=CHURN_PALETTE[0], fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df['Churn'] == 1], x='tenure', label=CHURN_LABELS[1], color=CHURN_PALETTE[1], fill=True, alpha=0.4, ax=ax)
    
    ax.set_title("Customer Tenure Distribution by Churn Status")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    
    save_plot(fig, "churn_by_tenure.png")

def plot_churn_by_monthly_charges(df):
    """
    Plots the density estimation of customer monthly charges split by churn.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    sns.kdeplot(data=df[df['Churn'] == 0], x='MonthlyCharges', label=CHURN_LABELS[0], color=CHURN_PALETTE[0], fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df['Churn'] == 1], x='MonthlyCharges', label=CHURN_LABELS[1], color=CHURN_PALETTE[1], fill=True, alpha=0.4, ax=ax)
    
    ax.set_title("Monthly Charges Distribution by Churn Status")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Density")
    ax.legend(title="Customer Status")
    
    save_plot(fig, "churn_by_charges.png")

def plot_churn_by_categorical_features(df):
    """
    Generates subplots showing Churn split by Internet Service and Payment Method.
    """
    # 1. Churn by Internet Service
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Internet Service cross-tab
    int_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    int_churn = int_churn.rename(columns=CHURN_LABELS)
    int_churn.plot(kind='bar', stacked=True, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], ax=axes[0], width=0.5)
    axes[0].set_title("Churn Rate by Internet Service")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("Internet Service")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend().remove()
    
    # Annotate Internet Service
    for p in axes[0].patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 5:
            axes[0].annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', color='white', fontweight='bold')

    # 2. Churn by Payment Method
    pay_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
    pay_churn = pay_churn.rename(columns=CHURN_LABELS)
    pay_churn.plot(kind='bar', stacked=True, color=[CHURN_PALETTE[0], CHURN_PALETTE[1]], ax=axes[1], width=0.5)
    axes[1].set_title("Churn Rate by Payment Method")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("Payment Method")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
    axes[1].legend(title="Customer Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Annotate Payment Method
    for p in axes[1].patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 5:
            axes[1].annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    save_plot(fig, "churn_by_services_and_payment.png")

def plot_correlation_matrix(df):
    """
    Computes and plots correlation matrix for numerical features.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    corr = df[numeric_cols].corr()
    
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Matrix of Numeric Features")
    
    save_plot(fig, "correlation_matrix.png")

def generate_all_eda_plots(df):
    """
    Generates and saves all EDA visualizations.
    """
    print("Generating exploratory data analysis plots...")
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_tenure(df)
    plot_churn_by_monthly_charges(df)
    plot_churn_by_categorical_features(df)
    plot_correlation_matrix(df)
    print("All EDA plots generated successfully.")
