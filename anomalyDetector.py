import numpy as np
import matplotlib
import terminalColors as tc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# See: https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
class AnomalyDetector:
    def __init__(self, outliers_fraction, window_size, file):
        self.outliers_fraction = outliers_fraction
        self.window_size = window_size
        self.stride = self.window_size / 2
        self.file = file
        # define outlier/anomaly detection methods to be compared
        self.anomaly_algorithms = [
            (
                "Robust covariance", 
                EllipticEnvelope(contamination=outliers_fraction)
            ),
            (
                "One-Class SVM", 
                svm.OneClassSVM(
                    nu=outliers_fraction, 
                    kernel="rbf",
                    gamma=0.1
                )
            ),
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

    def detect_anomalies(self, X, show=False):
        plot_num = 1
        plt.figure(figsize=(len(self.anomaly_algorithms) * 2 + 3, 6))

        for name, algorithm in self.anomaly_algorithms:
            tc.yellow('Detecting anomalies using {}...'.format(name))
            algorithm.fit(X)
            plt.subplot(1, len(self.anomaly_algorithms), plot_num)
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
        plt.savefig(self.file)
        tc.green('Saved anomaly plot to {}'.format(self.file))
        if show:
            plt.show()

    def plot_anomalies(self, name, X, y_pred, plt):
        # Create scatter plot
        colors = np.array(['#377eb8', '#ff7f00'])
        dim1 = 1  # first dimension to plot (of the features)
        dim2 = 2  # second dimension to plot (of the features)
        plt.scatter(X[:, dim1], X[:, dim2], s=10, color=colors[(y_pred + 1) // 2])
        for i, pred in enumerate(y_pred):
            if (pred < 0):
                plt.annotate('Index: {}'.format(i), (X[i, dim1], X[i, dim2]))

    def print_anomalies(self, name, y_pred):
        # Print to console
        print(name)
        for i, pred in enumerate(y_pred):
            if (pred == -1):
                print(
                    i, 
                    pred, 
                    i * self.stride, '-', 
                    i * self.stride + self.window_size
                )
