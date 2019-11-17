from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm
import umap


class DataProcessor:
    """This is where the dimensionality reduction happens.
    The timeseries in cut into smaller windows. On each window and ARMA model 
    is fittet. The parameters of the ARMA model serve as features.
    """
    def __init__(self, window_size, anomalies):
        self.anomalies = anomalies
        self.window_size = window_size
        self.stride = self.window_size / 2
        self.features = []
        self.window_labels = []

    def reduce_arma(self, timeseries, file_name):
        """Process the complete timeseries. Create windows first and then
        encode each window to reduce the dimensionality.

        Returns
        -------
        features: list
            List of features. 
        """
        windows = self.get_windows(timeseries)
        self.features = self.get_parameters(windows, self.get_arma_params)
        self.save_data(self.features, file_name)
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
    
    def save_data(self, data, file_name):
        """Write data to csv file.
        """
        df = pd.DataFrame(data) 
        file_path = 'data/{}.csv'.format(file_name)
        df.to_csv(file_path, header=False, index=False) 
        print('Saved features in {}.'.format(file_path))

    def print_features(self):
        for feature in self.features:
            print(feature)

    def visualize_features(self, features, file_name, method='TSNE'):
        print('Visualize features using {}...'.format(method))
        if (method == 'TSNE'):
            embedded = TSNE(n_components=2).fit_transform(features)
            self.plot_highlighted_anomalies(
                features,
                embedded,
                file_name,
                method
            )
        elif (method == 'UMAP'):
            reducer = umap.UMAP()
            embedded = reducer.fit_transform(features)
            self.plot_highlighted_anomalies(
                features,
                embedded,
                file_name,
                method
            )


    def plot_highlighted_anomalies(self, features, embedded, file_name, method):
        '''Add labels and color to anomaly datapoints
        '''
        fig = plt.figure()
        plt.scatter(embedded[:, 0], embedded[:, 1], c='blue')
        if (len(self.window_labels) < 1):
            start = 0
            end = self.window_size
            for i, feature in enumerate(features):
                self.window_labels.append([start, end])
                start = int( start + self.stride)
                end = int( end + self.stride)
        anomalies_str = ' Anomalies:'
        for anomaly in self.anomalies:
            start = anomaly * self.stride
            end = start + self.window_size
            for i, txt in enumerate(self.window_labels):
                if (txt == [start, end]):
                    plt.scatter(embedded[i, 0], embedded[i, 1], c='red')
                
                    anomalies_str +=  '(x={}, y={})\n'.format(
                        embedded[i, 0], 
                        embedded[i, 1]
                    )

                    plt.annotate(
                        'Anomaly: {}'.format(txt), 
                        (embedded[i, 0], embedded[i, 1])
                    )
        plt.title(
            '{} projection of the features\n {}'.format(method, anomalies_str)
        )
        fig.tight_layout()
        file_path = 'data/{}_{}-features.png'.format(file_name, method)
        fig.savefig(file_path)
        plt.show()
        plt.clf()
        print('Saved {} visualized features using {}'.format(method, file_path))