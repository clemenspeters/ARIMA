import random
import matplotlib.pyplot as plt
import numpy as np

def generate_multivariate_normal():
    mean = (1, 1)
    cov = ((1, 1.2), (1.2, 1))
    return np.random.multivariate_normal(mean, cov, 1000)

def plot_with_anomalies(X):
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c = 'blue')
    plt.scatter([-7, -8], [9, 8], c = 'red')
    # plt.title("Anomaly example")
    plt.xlabel("x")
    plt.ylabel("y");
    plt.show()
    fig.savefig('anomaly_plot.png')

multivariate_normal_data = generate_multivariate_normal()
plot_with_anomalies(multivariate_normal_data)
