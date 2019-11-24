from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm
import umap
from tqdm import tqdm
import terminalColors as tc


class DataProcessor:
    """This is where the dimensionality reduction happens.
    The timeseries in cut into smaller windows. On each window and ARMA model 
    is fittet. The parameters of the ARMA model serve as features.
    """

    def generate_features(self, timeseries, anomaly_labels, window_size, file_name, method='ARMA', stride=0):
        """Process the complete timeseries. Create windows first and then
        encode each window to reduce the dimensionality.

        Returns
        -------
        features: DataFrame
            List of features with anomaly labels
        """
        if stride == 0:
            stride = window_size / 2

        if (method == 'ARMA'):
            get_parameters = self.get_arma_params
        elif (method == 'ARIMA'):
            get_parameters = self.get_arima_params
        else: 
            raise ValueError(
                'Unkown method {}.'.format(method)
                + 'Only ARMA and ARIMA are supported.'
            )

        window_columns = ['window_start', 'window_end', 'is_anomaly']
        windows = pd.DataFrame(columns = window_columns) 

        features = pd.DataFrame() 
        window_starts = np.arange(0, len(timeseries), step=stride, dtype=int)
        tc.yellow("Generating features...")

        for i, start in enumerate(tqdm(window_starts)):
            end = int(start + window_size - 1)
            window_data = timeseries[start: end]
            window_is_anomaly = min(1, sum(anomaly_labels[start: end]))
            windows.loc[i] = [start, end, window_is_anomaly]

            fitted = get_parameters(window_data)
            if i == 0:
                feature_columns = np.append(fitted.data.param_names, ('is_anomaly', 'window_label'))
                features = pd.DataFrame(columns=feature_columns) 
            window_label = '{}-{}'.format(start, end)
            # TODO: add fitted.sigma2
            newRow = np.append(fitted.params, (window_is_anomaly, window_label))
            features.loc[i] = newRow

        features.is_anomaly = features.is_anomaly.astype(int)
        features.to_csv(file_name, index=False) # Save features to file
        tc.green('Saved features in {}'.format(file_name))
        return pd.read_csv(file_name)


    def get_arma_params(self, window_data):
        model = sm.tsa.ARMA(window_data, (2, 2))
        startParams=[.75, -.25, .65, .35] # Manual hack to avoid errors
        return model.fit(trend='nc', disp=0, start_params=startParams)

    # TODO: test arima params
    def get_arima_params(self, window_data):
        model = sm.tsa.ARIMA(window_data, (2, 1, 2))
        return model.fit(trend='nc', disp=0)

    def visualize_features(self, data, file_name, method='TSNE', show=False):
        fig = plt.figure()
        tc.yellow('Visualize features using {}...'.format(method))
        features = data.drop(['is_anomaly', 'window_label'], axis=1)

        if (method == 'TSNE'):
            embedded = TSNE(n_components=2).fit_transform(features)
        elif (method == 'UMAP'):
            embedded = umap.UMAP().fit_transform(features)

        ai = data.index[data.is_anomaly == 1].tolist()
        ni = data.index[data.is_anomaly == 0].tolist()

        normal = plt.scatter(embedded[ni, 0], embedded[ni, 1], c='blue')
        anomaly = plt.scatter(embedded[ai, 0], embedded[ai, 1], c='red')

        for i in ai:
            wl = data.loc[i].window_label
            plt.annotate(
                '{} ({})'.format(i, wl), 
                (embedded[i, 0], embedded[i, 1])
            )

        plt.legend((normal, anomaly), ('Normal', 'Anomaly'), loc='lower right')
        plt.title('{} projection of the features\n'.format(method))
        fig.tight_layout()
        file_path = '{}_{}-features.png'.format(file_name, method)
        fig.savefig(file_path)
        if show:
            plt.show()
        plt.close()
        tc.green(
            'Saved {} visualized features using {}'.format(method, file_path)
        )
