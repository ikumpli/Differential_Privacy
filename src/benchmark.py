import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from diffprivlib.models import LogisticRegression, GaussianNB
from diffprivlib.tools import mean, histogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import sklearn.naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DifferentialPrivacyBenchmark:
    def __init__(self, X, y, epsilon_values):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.epsilon_values = epsilon_values
        self.data_norm = 0
        self.results = []

    def scale_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.data_norm = np.linalg.norm(self.X_train, axis=1).max()

    def benchmark_private_model(self, model_name, model_class):
        # Iterate over epsilon values for private models
        for epsilon in self.epsilon_values:
            if model_class == LogisticRegression:
                model = LogisticRegression(epsilon=epsilon, data_norm=self.data_norm)
            elif model_class == GaussianNB:
                model = GaussianNB(epsilon=epsilon, bounds=(0, 1))

            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            metrics = {
                'Model': model_name,
                'Epsilon': epsilon,
                'Accuracy': accuracy_score(self.y_test, predictions),
                'Precision': precision_score(self.y_test, predictions),
                'Recall': recall_score(self.y_test, predictions),
                'F1': f1_score(self.y_test, predictions)
            }
            self.results.append(metrics)

    def benchmark_non_private_model(self, model_name, model_class):
        # Non-private models are benchmarked only once
        if model_class == LogisticRegression:
            model = SklearnLogisticRegression()
        elif model_class == sklearn.naive_bayes.GaussianNB:
            model = sklearn.naive_bayes.GaussianNB()

        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        metrics = {
            'Model': model_name,
            'Epsilon': 'Non-Private',
            'Accuracy': accuracy_score(self.y_test, predictions),
            'Precision': precision_score(self.y_test, predictions),
            'Recall': recall_score(self.y_test, predictions),
            'F1': f1_score(self.y_test, predictions)
        }
        self.results.append(metrics)

    def get_results_dataframe(self):
        return pd.DataFrame(self.results)

    def plot_metrics(self):
        df = self.get_results_dataframe()
        
        # Separate private and non-private results
        private_df = df[df['Epsilon'] != 'Non-Private']
        non_private_df = df[df['Epsilon'] == 'Non-Private']
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        models = df['Model'].unique()
        colors = ['blue', 'orange', 'green', 'red']  # Define distinct colors for each model
        
        # Create subplots for each metric
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        for i, metric in enumerate(metrics):
            ax = axs[i]
            for j, model in enumerate(models):
                model_private_df = private_df[private_df['Model'] == model]
                color = colors[j % len(colors)]  # Assign a color to the model

                # Plot private results
                ax.plot(
                    model_private_df['Epsilon'],
                    model_private_df[metric],
                    marker='o',
                    label=f"{model} (Private)",
                    color=color
                )

                # Plot non-private results as dashed lines
                if not non_private_df[non_private_df['Model'] == model].empty:
                    non_private_value = non_private_df[non_private_df['Model'] == model][metric].values[0]
                    ax.axhline(
                        y=non_private_value,
                        color=color,
                        linestyle='--',
                        label=f"{model} (Non-Private)"
                    )

            ax.set_xscale('log')
            ax.set_xlabel("Epsilon")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Epsilon")
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.suptitle("Differential Privacy Benchmark Metrics", fontsize=16, y=1.02)
        plt.show()
