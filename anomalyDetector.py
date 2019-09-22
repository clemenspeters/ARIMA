import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


outliers_fraction = 0.002
# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    (
        "Isolation Forest",
        IsolationForest(
            behaviour='new',
            contamination=outliers_fraction,
            random_state=42
        )
    ),
    (
        "Local Outlier Factor", 
        LocalOutlierFactor(
            n_neighbors=35, 
            contamination=outliers_fraction
        )
    )
]

class AnomalyDetector:
# See: https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py

    def detect_anomalies(self, X):
        plot_num = 1
        plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 6))

        for name, algorithm in anomaly_algorithms:
            algorithm.fit(X)
            plt.subplot(1, len(anomaly_algorithms), plot_num)
            plt.title(name, size=18)

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)
            # Print and plot
            self.print_anomalies(name, y_pred)
            self.plot_anomalies(name, X, y_pred, plt)
            plot_num += 1
        plt.tight_layout()
        plt.savefig('img/anomalies.png')
        plt.show()

    def plot_anomalies(self, name, X, y_pred, plt):
        # Create scatter plot
        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 2], X[:, 3], s=10, color=colors[(y_pred + 1) // 2])
        # plt.xlim(-7, 7)
        # plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, name,
            transform=plt.gca().transAxes, size=15,
            horizontalalignment='right')
        for i, pred in enumerate(y_pred):
            if (pred < 0):
                plt.annotate('Index: {}'.format(i), (X[i, 2], X[i, 3]))

    def print_anomalies(self, name, y_pred):
        # Print to console
        print(name)
        for i, pred in enumerate(y_pred):
            if (pred == -1):
                print(i, pred)
