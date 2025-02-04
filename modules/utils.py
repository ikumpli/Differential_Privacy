import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from diffprivlib.models import (LogisticRegression, GaussianNB,
                                RandomForestClassifier, DecisionTreeClassifier)
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import ipywidgets as widgets
from IPython.display import display


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

class DifferentialPrivacyBenchmark:
    def __init__(self, X, y, epsilon_values):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.epsilon_values = epsilon_values
        self.data_norm = 0
        self.results = []

    def scale_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.data_norm = np.linalg.norm(self.X_train, axis=1).max()
        
        self.bounds_X = (float(np.min(self.X_train)), float(np.max(self.X_train)))
        self.bounds_y = (float(np.min(self.y_train)), float(np.max(self.y_train)))

    def benchmark_private_model(self, model_name, model_class):
        for epsilon in self.epsilon_values:
            if model_class == LogisticRegression:
                model = LogisticRegression(epsilon=epsilon, data_norm=self.data_norm)
            elif model_class == GaussianNB:
                model = GaussianNB(epsilon=epsilon, bounds=(0,1))
            elif model_class == RandomForestClassifier:
                model = RandomForestClassifier(epsilon=epsilon, classes=[0, 1], bounds=(0,1))
            elif model_class == DecisionTreeClassifier:
                model = DecisionTreeClassifier(epsilon=epsilon, classes=[0, 1], bounds=(0,1))

            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            metrics = self._compute_metrics(predictions, model_name, epsilon)
            self.results.append(metrics)

    def benchmark_non_private_model(self, model_name, model_class):
        if model_class == SklearnLogisticRegression:
            model = SklearnLogisticRegression()
        elif model_class == SklearnGaussianNB:
            model = SklearnGaussianNB()
        elif model_class == SklearnRandomForestClassifier:
            model = SklearnRandomForestClassifier()
        elif model_class == SklearnDecisionTreeClassifier:
            model = SklearnDecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        metrics = self._compute_metrics(predictions, model_name, 'Non-Private')
        self.results.append(metrics)

    def _compute_metrics(self, predictions, model_name, epsilon):
        return {
            'Model': model_name,
            'Epsilon': epsilon,
            'Accuracy': accuracy_score(self.y_test, predictions),
            'Precision': precision_score(self.y_test, predictions, zero_division=0),
            'Recall': recall_score(self.y_test, predictions),
            'F1': f1_score(self.y_test, predictions)
        }
        
    def get_results_dataframe(self):
        return pd.DataFrame(self.results)

    def plot_metrics(self, save_img_path=None):
        df = self.get_results_dataframe()

        # Select metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        # Separate private and non-private results
        private_df = df[df['Epsilon'] != 'Non-Private']
        non_private_df = df[df['Epsilon'] == 'Non-Private']

        models = df['Model'].unique()
        colors = ['blue', 'orange', 'green', 'red']

        num_metrics = len(metrics)
        cols = 2
        rows = (num_metrics + 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(10, 6))
        axs = axs.flatten()  # Flatten subplots

        # Store legend handles
        legend_handles = []

        for i, metric in enumerate(metrics):
            ax = axs[i]
            for j, model in enumerate(models):
                model_private_df = private_df[private_df['Model'] == model]
                color = colors[j % len(colors)]

                # Plot private results (solid line)
                line_private, = ax.plot(
                    model_private_df['Epsilon'],
                    model_private_df[metric],
                    marker='o',
                    label=f"{model} (Private)",
                    color=color
                )
                if f"{model} (Private)" not in [h.get_label() for h in legend_handles]:
                    legend_handles.append(line_private)

                # Plot non-private baseline as dashed line
                model_nonprivate_df = non_private_df[non_private_df['Model'] == model]
                if not model_nonprivate_df.empty:
                    non_private_value = model_nonprivate_df[metric].values[0]
                    line_non_private = ax.axhline(
                        y=non_private_value,
                        color=color,
                        linestyle='--',
                        label=f"{model} (Non-Private)"
                    )
                    if f"{model} (Non-Private)" not in [h.get_label() for h in legend_handles]:
                        legend_handles.append(line_non_private)

            ax.set_xscale('log')
            ax.set_xlabel("Epsilon")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Epsilon")
            ax.grid()

        # Consolidated legend
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.9),
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.suptitle("Differential Privacy Benchmark Metrics", fontsize=16, y=0.94)
        if save_img_path==None:
            plt.savefig("benchmark_metrics.png", dpi=300)
        else:
            plt.savefig(save_img_path, dpi=300)
        plt.show()

    def plot_interactive_bar_chart(self):
        """
        Creates an interactive bar chart with the ability to toggle between metrics
        (e.g., F1 Score, Accuracy) and adjust epsilon using a slider.
        The slider now uses discrete indices corresponding to the predefined epsilon values,
        and the dropdown allows selecting the metric.
        """
        df = self.get_results_dataframe()

        # Separate private and non-private rows
        df_private = df[df['Epsilon'] != 'Non-Private']
        df_nonprivate = df[df['Epsilon'] == 'Non-Private']

        # Get the private epsilon values (assumed to be numerical)
        private_epsilons = np.array(sorted(df_private['Epsilon'].unique()))
        models = df_private['Model'].unique()

        # Available metrics for selection
        metrics = ['F1', 'Accuracy', 'Precision', 'Recall']

        # Initial settings
        initial_metric = metrics[0]
        initial_index = 0
        initial_epsilon = private_epsilons[initial_index]
        filtered_df = df_private[df_private['Epsilon'] == initial_epsilon]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)  # Adjust to fit the slider

        # Create bars for each model
        bars = ax.bar(models, filtered_df[initial_metric], color='blue')
        ax.set_ylim(0, 1)
        ax.set_title(f"{initial_metric} Comparison (Epsilon={initial_epsilon:.2f})")
        ax.set_xlabel("Models")
        ax.set_ylabel(initial_metric)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Store references to baseline lines
        baseline_lines = []

        # Function to add baseline lines
        def add_baseline_lines(current_metric):
            # Remove existing lines
            for line in baseline_lines:
                line.remove()
            baseline_lines.clear()

            # Add new baseline lines
            for i, model_name in enumerate(models):
                row = df_nonprivate[df_nonprivate['Model'] == model_name]
                if not row.empty:
                    baseline_metric = row[current_metric].values[0]
                    bar_x = bars[i].get_x() + bars[i].get_width() / 2
                    half_width = bars[i].get_width() / 2
                    line = ax.hlines(
                        y=baseline_metric,
                        xmin=bar_x - half_width * 0.5,
                        xmax=bar_x + half_width * 0.5,
                        colors='red',
                        linestyles='dotted',
                        label=("Non-Private Baseline" if i == 0 else None)
                    )
                    baseline_lines.append(line)
            ax.legend()

        # Add initial baseline lines
        add_baseline_lines(initial_metric)

        # Create the slider using ipywidgets
        slider = widgets.IntSlider(
            value=initial_index,
            min=0,
            max=len(private_epsilons) - 1,
            step=1,
            description="Epsilon",
            style={"description_width": "initial"},
        )

        # Create the dropdown using ipywidgets
        dropdown = widgets.Dropdown(
            options=metrics,
            value=initial_metric,
            description="Metric",
            style={"description_width": "initial"},
        )

        # Update function for the slider and dropdown
        def update_plot(change):
            current_index = slider.value
            current_epsilon = private_epsilons[current_index]
            current_metric = dropdown.value

            # Filter data for the selected epsilon
            filtered = df_private[df_private['Epsilon'] == current_epsilon]
            for bar_rect, new_value in zip(bars, filtered[current_metric]):
                bar_rect.set_height(new_value)

            # Update title and baseline lines
            ax.set_title(f"{current_metric} Comparison (Epsilon={current_epsilon:.2f})")
            ax.set_ylabel(current_metric)
            ax.set_ylim(0, 1)

            # Re-add baseline lines for the selected metric
            add_baseline_lines(current_metric)
            fig.canvas.draw_idle()

        # Connect the widgets to the update function
        slider.observe(update_plot, names='value')
        dropdown.observe(update_plot, names='value')

        # Display the widgets below the plot
        display(slider, dropdown)
        plt.show()
