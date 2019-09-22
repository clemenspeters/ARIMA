import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

outliers_fraction = 0.002
# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42))]

class AnomalyDetector:
# See: https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py

    def detect_anomalies(self, X):
        plot_num = 1
        plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))

        for name, algorithm in anomaly_algorithms:
                algorithm.fit(X)
                plt.subplot(1, len(anomaly_algorithms), plot_num)
                plt.title(name, size=18)

                # fit the data and tag outliers
                if name == "Local Outlier Factor":
                    y_pred = algorithm.fit_predict(X)
                else:
                    y_pred = algorithm.fit(X).predict(X)
        self.print_anomalies(y_pred)

    def print_anomalies(self, y_pred):
        for i, pred in enumerate(y_pred):
            if (pred == -1):
                print(i, pred)