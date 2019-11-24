# Source: https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_process.html#arma_generate_sample
from scipy import signal
import numpy as np
import pandas as pd
import statsmodels.tsa.arima_process as arima
import statsmodels.tsa as sm
import terminalColors as tc
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class DataGenerator:
    """
    Generates a time series which consits of number of windows.
    Some of these windows contain anomalies.

    Parameters
    ----------
    window_count (int)
        Number of windows in generated series.\n
    window_size : int
        Number of observations/samples per window.\n
    anomalies : array
        Indices of the windows containing anomalies.\n
    """
    def __init__(self, window_count, window_size, anomalies, file_name):
        self.anomalies = anomalies
        self.window_size = window_size
        self.window_count = window_count
        self.nsample = window_count * window_size  # number of observations/samples
        self.file_name = file_name

    def generate_timeseries(self, show=True, seed=12345):
        np.random.seed(seed)
        """Stitch together two time series with different  ARMA parameters
        to generate one timeseries which contains anomalies.

        Parameters
        ----------
        show : bool
            Show generated data as plots.
        
        Returns
        -------
        stitched_data: array
            Data containing anomalies.
        """
        # Genrate the two timeseries (with different ARMA parameters)
        tc.yellow('Generating normal timeseries...')
        ar, ma = self.arma_generate_params([.75, -.25], [.65, .35])
        default_series = arima.arma_generate_sample(ar, ma, self.nsample)
        default_series = pd.DataFrame(default_series, columns=['value'])
        default_series['is_anomaly'] = int(0)

        tc.yellow('Generating anomaly timeseries...')
        ar, ma = self.arma_generate_params([.75, -.25], [-.65, .35])
        anomaly_series = arima.arma_generate_sample(ar, ma, self.nsample)
        anomaly_series = pd.DataFrame(anomaly_series, columns=['value'])
        anomaly_series['is_anomaly'] = int(1)

        # Plot the two timeseries
        if show:
            self.show_raw_data(default_series, anomaly_series)
        
        tc.yellow(
            'Combining the two timeseries to get one time series'
            'containing anomalies...'
        )
        stitched_data = default_series
        for anomaly in self.anomalies:
            start = anomaly * self.window_size
            end = (anomaly + 1) * self.window_size
            # Inject anomalies
            stitched_data[start : end] =  anomaly_series[start : end]

        self.create_data_plot(stitched_data, show)

        self.save_data(stitched_data)
        return pd.DataFrame(stitched_data) 

    def arma_generate_sample(self, ar, ma):
        """
        See: https://www.statsmodels.org/devel/_modules/statsmodels/tsa/arima_process.html#ArmaProcess.generate_sample
        """
        arparams = np.array(ar)
        maparams = np.array(ma)
        arparams = np.r_[1, -arparams] # add zero-lag and negate
        maparams = np.r_[1, maparams]  # add zero-lag
        # Generate samples
        sigma = 1                               # standard deviation of noise
        distrvs = np.random.randn               # function that generates the random numbers, and takes sample size as argument
        eta = sigma * distrvs(self.nsample)     # this is where the random samples are drawn. (((Maybe we can insert our anomalies here?)))
        return signal.lfilter(maparams, arparams, eta, axis=0)

    def arma_generate_params(self, ar, ma):
        """
        See: https://www.statsmodels.org/devel/_modules/statsmodels/tsa/arima_process.html#ArmaProcess.generate_sample
        """
        arparams = np.array(ar)
        maparams = np.array(ma)
        arparams = np.r_[1, -arparams] # add zero-lag and negate
        maparams = np.r_[1, maparams]  # add zero-lag
        return arparams, maparams
    
    def save_data(self, data):
        """Write data to pandas csv file.
        """
        df = pd.DataFrame(data) 
        file_path = '{}.csv'.format(self.file_name)
        df.to_csv(file_path, index=False) 
        tc.green('Saved data in {}.'.format(file_path))

    def create_data_plot(self, data, show):
        """Plot the generated (stitched) data containing the anomalies.
        """
        fig = plt.figure()
        plt.title(self.get_title())
        cmap = ['b', 'r']
        # plt.plot(data.mask((data['is_anomaly'] == 1))['value'], color='blue')
        plt.plot(data['value'], color='blue')
        plt.plot(data.mask((data['is_anomaly'] == 0))['value'], color='red')
        plt.tight_layout() # avoid overlapping plot titles
        file_path = '{}.png'.format(self.file_name)
        fig.savefig(file_path)
        tc.green('Saved data plot in {}'.format(file_path))
        if show:
            plt.show()
        plt.close()

    def get_title(self):
        """
        Generate title for the data plot containing all anomalies.
        """
        title = 'Generated training data.\n{} windows stitched together.\nWindow size = {}.'.format(
            self.window_count, 
            self.window_size,
        )
        anomaliesStr = ' Anomalies:'
        for anomaly in self.anomalies:
            start = anomaly * self.window_size
            end = (anomaly + 1) * self.window_size
            anomaliesStr +=  '\n({}:{})'.format(start, end)
        return title + anomaliesStr

    def show_raw_data(self, default_series, anomaly_series):
        """Plot the two time series which will be stitched together.
        """
        fig = plt.figure(1, figsize=(12, 3))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.title.set_text('Normal time series.')
        ax2.title.set_text('Anomaly time series.')
        ax1.plot(np.arange(self.nsample), default_series, c='blue')
        ax2.plot(np.arange(self.nsample), anomaly_series, c='red')
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('img/raw_data.png')
        plt.show()
        plt.close()