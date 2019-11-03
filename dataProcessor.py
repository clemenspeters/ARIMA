from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.style.use('ggplot')


class DataProcessor:
    """This is where the dimensionality reduction happens.
    The timeseries in cut into smaller windows. On each window and ARMA model 
    is fittet. The parameters of the ARMA model serve as features.
    """
    def __init__(self, window_size, anomalies, data_file = 'features'):
        self.anomalies = anomalies
        self.window_size = window_size
        self.stride = self.window_size / 2
        self.features = []
        self.window_labels = []
        self.data_file = data_file

    def reduce_arma(self, timeseries):
        """Process the complete timeseries. Create windows first and then
        encode each window to reduce the dimensionality.

        Returns
        -------
        features: list
            List of features. 
        """
        windows = self.get_windows(timeseries)
        self.features = self.get_parameters(windows, self.get_arma_params)
        self.save_data(self.features)
        return self.features

    def get_windows(self, timeseries):
        """Cuts the given timeseries in windows and creates a list of labels
        for the crated windows (self.window_labels).

        Parameters
        ----------
        timeseries: array-like
            Original timeseries data which is cut into windows.

        Returns
        -------
        windows: list
            List of windows. 
        """
        windowList = []
        start = 0
        end = self.window_size
        while (end <= timeseries.size):
            self.window_labels.append([start, end])
            windowList.append(timeseries[start: end])
            start = int( start + self.stride)
            end = int( end + self.stride)
        return windowList

    def get_parameters(self, windows, encoder_function):
        """Iterates over all windows and runs the encoder_function on 
        each window.
        """
        parametersList = []
        for window in windows:
            params = encoder_function(window)
            parametersList.append(params)
        return parametersList

    def get_arma_params(self, dataWindow):
        model = sm.tsa.ARMA(dataWindow, (2, 2))
        startParams=[.75, -.25, .65, .35] # Manual hack to avoid errors
        result = model.fit(trend='nc', disp=0, start_params=startParams)
        # result = model.fit(trend='nc', disp=0)
        return result.params
    
    def save_data(self, data):
        """Write data to features.npy file.
        """
        np.save('results/{}'.format(self.data_file), data)

    def load_data(self):
        """Load data from features.npy file.
        """
        return np.load('results/{}.npy'.format(self.data_file))

    def print_features(self):
        for feature in self.features:
            print(feature)

    def visualize_features(self, features, method='TSNE'):
        if (method == 'TSNE'):
            embedded = TSNE(n_components=2).fit_transform(features)
            fig = plt.figure(1, figsize=(12, 3))
            sub1 = fig.add_subplot(111)
            sub1.scatter(embedded[:, 0], embedded[:, 1])
             # Add labels and color to anomaly datapoints
            if (len(self.window_labels) < 1):
                start = 0
                end = self.window_size
                for i, feature in enumerate(features):
                    self.window_labels.append([start, end])
                    start = int( start + self.stride)
                    end = int( end + self.stride)
            anomaliesStr = ' Anomalies:'
            for anomaly in self.anomalies:
                start = anomaly * self.stride
                end = start + self.window_size
                for i, txt in enumerate(self.window_labels):
                    if (txt == [start, end]):
                        print(i)
                        sub1.scatter(embedded[i, 0], embedded[i, 1], c='green')
                        anomaliesStr +=  '(x={}, y={})\n'.format(embedded[i, 0], embedded[i, 1])
                        sub1.annotate('Anomaly: {}'.format(txt), (embedded[i, 0], embedded[i, 1]))
            sub1.title.set_text('TSNE embedded features\n. {}'.format(anomaliesStr))
            fig.tight_layout()
            fig.savefig('img/TSNE-features.png')
            # Show the plot in non-blocking mode
            plt.show()

