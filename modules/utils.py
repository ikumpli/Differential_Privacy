import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diffprivlib.tools import mean, histogram
from matplotlib.lines import Line2D

COLOR_PALETTE = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

def load_and_preprocess_titanic():
    """
    Loads the Titanic dataset from a GitHub URL and performs basic
    preprocessing like filling missing values and encoding categorical data.

    Returns
    -------
    X : pd.DataFrame
        The feature matrix.
    y : pd.Series
        The target vector (Survived).
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    X = df[features]
    y = df[target]
    return X, y

def calculate_private_statistics(df, epsilon_values):
    """
    Calculate private statistics for Age, Survival Rate, and Class Distributions
    for a given set of epsilon values.
    """
    age_bounds = (0, 80)
    survived_bounds = (0, 1)
    class_range = (1, 3)

    private_mean_ages = []
    private_survival_rates = []
    private_class_distributions = []

    for epsilon in epsilon_values:
        # Calculate private mean age
        private_mean_ages.append(mean(df['Age'].dropna(), epsilon=epsilon, bounds=age_bounds))
        # Calculate private survival rate
        private_survival_rates.append(mean(df['Survived'], epsilon=epsilon, bounds=survived_bounds))
        # Calculate private class distribution
        private_class_counts, _ = histogram(df['Pclass'], epsilon=epsilon, range=class_range, bins=3)
        private_class_distributions.append(private_class_counts / private_class_counts.sum())

    return private_mean_ages, private_survival_rates, private_class_distributions

def plot_private_statistics(epsilon_values, original_stats, private_stats):
    """
    Plot comparison of original vs private stats (Age, Survival Rate, Class).
    With consistent styling for figure/axes.
    """
    original_mean_age, original_survival_rate, original_class_counts = original_stats
    private_mean_ages, private_survival_rates, private_class_distributions = private_stats

    # We'll choose a width that comfortably fits 3 subplots in 1 row
    plt.figure(figsize=(12, 4.5))

    # --- Mean Age ---
    plt.subplot(1, 3, 1)
    plt.plot(epsilon_values, private_mean_ages, marker='o', label="Private Mean Age")
    plt.axhline(original_mean_age, color='red', linestyle='--', label="Original Mean Age")
    plt.title("Mean Age vs Epsilon", fontsize=12)
    plt.xlabel("Epsilon", fontsize=10)
    plt.ylabel("Mean Age", fontsize=10)
    plt.grid(True)
    plt.legend()

    # --- Survival Rate ---
    plt.subplot(1, 3, 2)
    plt.plot(epsilon_values, private_survival_rates, marker='o', label="Private Survival Rate")
    plt.axhline(original_survival_rate, color='red', linestyle='--', label="Original Survival Rate")
    plt.title("Survival Rate vs Epsilon", fontsize=12)
    plt.xlabel("Epsilon", fontsize=10)
    plt.ylabel("Survival Rate", fontsize=10)
    plt.grid(True)
    plt.legend()

    # --- Class Distribution ---
    plt.subplot(1, 3, 3)
    for i in range(3):  # e.g., classes 1, 2, 3
        plt.plot(
            epsilon_values, 
            [dist[i] for dist in private_class_distributions], 
            marker='o',
            label=f"Class {i+1}"
        )
    # Original lines
    plt.axhline(original_class_counts[1], color='red', linestyle='--',
                label="Original Class 1", alpha=0.5)
    plt.axhline(original_class_counts[2], color='green', linestyle='--',
                label="Original Class 2", alpha=0.5)
    plt.axhline(original_class_counts[3], color='blue', linestyle='--',
                label="Original Class 3", alpha=0.5)
    plt.title("Class Distribution vs Epsilon", fontsize=12)
    plt.xlabel("Epsilon", fontsize=10)
    plt.ylabel("Normalized Class Proportion", fontsize=10)
    plt.grid(True)
    plt.legend()

    plt.suptitle("Private Statistics Comparison", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_dp_metrics(private_metrics_df, non_private_metrics, img_path=None, color_palette=COLOR_PALETTE):
    """
    Plots Accuracy, Precision, Recall, and F1 vs Epsilon (log scale)
    with a single consolidated legend and consistent styling.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # Unpack baseline
    baseline_acc = non_private_metrics['accuracy']
    baseline_prec = non_private_metrics['precision']
    baseline_recall = non_private_metrics['recall']
    baseline_f1 = non_private_metrics['f1']

    # Accuracy
    axs[0, 0].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['accuracy'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[0]
    )
    axs[0, 0].axhline(
        y=baseline_acc, 
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[0, 0].set_title("Accuracy", fontsize=12)
    axs[0, 0].set_xlabel("Epsilon", fontsize=10)
    axs[0, 0].set_ylabel("Metric Value", fontsize=10)
    axs[0, 0].set_ylim(0.0, 1.0)
    axs[0, 0].set_xscale("log")
    axs[0, 0].grid()

    # Precision
    axs[0, 1].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['precision'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[1]
    )
    axs[0, 1].axhline(
        y=baseline_prec,
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[0, 1].set_title("Precision", fontsize=12)
    axs[0, 1].set_xlabel("Epsilon", fontsize=10)
    axs[0, 1].set_ylabel("Metric Value", fontsize=10)
    axs[0, 1].set_ylim(0.0, 1.0)
    axs[0, 1].set_xscale("log")
    axs[0, 1].grid()

    # Recall
    axs[1, 0].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['recall'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[2]
    )
    axs[1, 0].axhline(
        y=baseline_recall,
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[1, 0].set_title("Recall", fontsize=12)
    axs[1, 0].set_xlabel("Epsilon", fontsize=10)
    axs[1, 0].set_ylabel("Metric Value", fontsize=10)
    axs[1, 0].set_ylim(0.0, 1.0)
    axs[1, 0].set_xscale("log")
    axs[1, 0].grid()

    # F1 Score
    axs[1, 1].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['f1'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[3]
    )
    axs[1, 1].axhline(
        y=baseline_f1,
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[1, 1].set_title("F1 Score", fontsize=12)
    axs[1, 1].set_xlabel("Epsilon", fontsize=10)
    axs[1, 1].set_ylabel("Metric Value", fontsize=10)
    axs[1, 1].set_ylim(0.0, 1.0)
    axs[1, 1].set_xscale("log")
    axs[1, 1].grid()

    # Consolidated legend handles
    legend_handles = [
        Line2D([0], [0], color='red', linestyle='--', label='Non-Private Baseline'),
        Line2D([0], [0], color=color_palette[0], marker='o', linestyle='-', label='DP Accuracy'),
        Line2D([0], [0], color=color_palette[1], marker='o', linestyle='-', label='DP Precision'),
        Line2D([0], [0], color=color_palette[2], marker='o', linestyle='-', label='DP Recall'),
        Line2D([0], [0], color=color_palette[3], marker='o', linestyle='-', label='DP F1 Score')
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
        prop={'size': 8}
    )

    plt.suptitle("Differentially Private Logistic Regression Metrics", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if img_path==None:
        plt.savefig('dp_metrics.png', dpi=300)
    else:
        plt.savefig(img_path, dpi=300)
    plt.show()

def plot_dp_metrics_acc_f1(private_metrics_df, non_private_metrics, img_path=None, color_palette=COLOR_PALETTE):
    """
    Plots only Accuracy and F1 Score vs Epsilon (log scale) in a 2-row, 1-column layout.
    Displays a consolidated legend at the top with the DP curves and the non-private baseline.
    """
    # Create 2 rows x 1 column layout
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    
    # Unpack baseline values
    baseline_acc = non_private_metrics['accuracy']
    baseline_f1  = non_private_metrics['f1']
    
    # --- Accuracy subplot ---
    axs[0].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['accuracy'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[0]
    )
    axs[0].axhline(
        y=baseline_acc,
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[0].set_title("Accuracy", fontsize=12)
    axs[0].set_xlabel("Epsilon", fontsize=10)
    axs[0].set_ylabel("Metric Value", fontsize=10)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_xscale("log")
    axs[0].grid()

    # --- F1 Score subplot ---
    axs[1].plot(
        private_metrics_df['epsilon'],
        private_metrics_df['f1'],
        marker='o',
        markersize=3.5,
        linewidth=1.3,
        color=color_palette[3]
    )
    axs[1].axhline(
        y=baseline_f1,
        color='red',
        linestyle='--',
        linewidth=1.3
    )
    axs[1].set_title("F1 Score", fontsize=12)
    axs[1].set_xlabel("Epsilon", fontsize=10)
    axs[1].set_ylabel("Metric Value", fontsize=10)
    axs[1].set_ylim(0.0, 1.0)
    axs[1].set_xscale("log")
    axs[1].grid()
    
    # Consolidated legend handles
    legend_handles = [
        Line2D([0], [0], color='red', linestyle='--', label='Non-Private Baseline'),
        Line2D([0], [0], color=color_palette[0], marker='o', linestyle='-', label='DP Accuracy'),
        Line2D([0], [0], color=color_palette[3], marker='o', linestyle='-', label='DP F1 Score')
    ]
    
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.92),
        frameon=False,
        prop={'size': 8}
    )
    
    plt.suptitle("Differentially Private Logistic Regression Metrics", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    if img_path is None:
        plt.savefig('dp_metrics_acc_f1.png', dpi=300)
    else:
        plt.savefig(img_path, dpi=300)
    plt.show()

def plot_f1_score_comparison(private_metrics_df, non_private_metrics, img_path=None, color_palette=COLOR_PALETTE):
    """
    Creates bins of epsilon (log-spaced), computes average & std of F1,
    and plots a bar chart for DP results with error bars.
    Adds a dashed red horizontal line for non-private F1 baseline.
    """
    # Define bins
    epsilon_bins = np.logspace(-2, 4, 10)
    
    # Create range labels using fixed-point formatting (avoid scientific notation)
    range_labels = [
        f"{epsilon_bins[i]:.2f} - {epsilon_bins[i + 1]:.2f}"
        for i in range(len(epsilon_bins) - 1)
    ]

    f1_avg, f1_std = [], []

    for i in range(len(epsilon_bins) - 1):
        mask = ((private_metrics_df['epsilon'] >= epsilon_bins[i]) &
                (private_metrics_df['epsilon'] < epsilon_bins[i + 1]))
        avg_f1 = private_metrics_df.loc[mask, 'f1'].mean()
        std_f1 = private_metrics_df.loc[mask, 'f1'].std()
        f1_avg.append(avg_f1)
        f1_std.append(std_f1)

    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    x = np.arange(len(range_labels))

    plt.bar(
        x, f1_avg, bar_width, yerr=f1_std,
        label="DP F1 (Averaged)",
        color=color_palette[0],
        capsize=5
    )

    baseline_f1 = non_private_metrics['f1']
    plt.axhline(
        y=baseline_f1,
        color='red',
        linestyle='--',
        label='Non-Private Baseline'
    )

    plt.xlabel("Epsilon Ranges (Log Scale)", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0.0, 1.0)
    plt.title("F1 Score Comparison: Non-Private vs Differentially Private Models", fontsize=12)
    plt.xticks(x, range_labels, rotation=45)
    plt.legend(prop={'size': 8})
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if img_path is None:
        plt.savefig('f1_comparison.png', dpi=300)
    else:
        plt.savefig(img_path, dpi=300)
    plt.show()