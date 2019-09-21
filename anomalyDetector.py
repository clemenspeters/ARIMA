from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.style.use('ggplot')


class AnomalyDetector:
    def __init__(self, window_size, anomalies):
        self.anomalies = anomalies
        self.window_size = window_size
        self.stride = self.window_size / 2
        self.features = []
        self.window_labels = []

    def reduce_arma(self, data):
        windows = self.get_windows(data)
        self.features = self.get_parameters(windows, self.get_arma_params)
        self.save_data(self.features)

    def get_windows(self, data):
        result = []
        start = 0
        end = self.window_size
        while (end <= data.size):
            self.window_labels.append([start, end])
            result.append(data[start: end])
            start = int( start + self.stride)
            end = int( end + self.stride)
        return result

    def get_parameters(self, windows, encoder_function):
        parameters = []
        for window in windows:
            params = encoder_function(window)
            parameters.append(params)
        return parameters

    def get_arma_params(self, dataWindow):
        model = sm.tsa.ARMA(dataWindow, (2, 2))
        startParams=[.75, -.25, .65, .35] # Manual hack to avoid errors
        result = model.fit(trend='nc', disp=0, start_params=startParams)
        # result = model.fit(trend='nc', disp=0)
        return result.params
    
    def save_data(self, data):
        """Write data to features.npy file.
        """
        np.save('features', data)

    def load_data(self):
        """Load data from features.npy file.
        """
        return np.load('features.npy')

    def print_features(self):
        for feature in self.features:
            print(feature)

    def visualize_features(self, method='TSNE'):
        if (method == 'TSNE'):
            embedded = TSNE(n_components=2).fit_transform(self.features)
            fig = plt.figure(1, figsize=(12, 3))
            sub1 = fig.add_subplot(111)
            sub1.scatter(embedded[:, 0], embedded[:, 1])
             # Add labels and color to anomaly datapoints
            for anomaly in self.anomalies:
                start = anomaly * self.window_size
                end = (anomaly + 1) * self.window_size
                for i, txt in enumerate(self.window_labels):
                    if (txt == [start, end]):
                        sub1.scatter(embedded[i, 0], embedded[i, 1], c='green')
                        sub1.annotate('Anomaly: {}'.format(txt), (embedded[i, 0], embedded[i, 1]))
            fig.savefig('TSNE-features.png')
            # Show the plot in non-blocking mode
            plt.show()

