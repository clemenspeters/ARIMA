# Source: https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_process.html#arma_generate_sample
from scipy import signal
import numpy as np
import pandas as pd
import statsmodels.tsa.arima_process as arima
import statsmodels.tsa as sm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(12345)

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
    def __init__(self, window_count, window_size, anomalies):
        self.anomalies = anomalies
        self.window_size = window_size
        self.window_count = window_count
        self.nsample = window_count * window_size  # number of observations/samples

    def generate_data(self, plot_data=True, file_name='generated_data'):
        """Stitch together two time series with different  ARMA parameters
        to generate one timeseries which contains anomalies.

        Parameters
        ----------
        plot_data : bool
            Show generated data as plots.
        
        Returns
        -------
        stitched_data: array
            Data containing anomalies.
        """
        # # Genrate the two timeseries (with different ARMA parameters)
        ar, ma = self.arma_generate_params([.75, -.25], [.65, .35])
        default_series = arima.arma_generate_sample(ar, ma, self.nsample)

        ar, ma = self.arma_generate_params([.75, -.25], [-.65, .35])
        anomaly_series = arima.arma_generate_sample(ar, ma, self.nsample)
        # Plot the two timeseries
        if plot_data:
            self.show_raw_data(default_series, anomaly_series)
        # Combine the two timeseries to get one time series containing anomalies
        stitched_data = default_series
        for anomaly in self.anomalies:
            start = anomaly * self.window_size
            end = (anomaly + 1) * self.window_size
            # Inject anomalies
            stitched_data[start : end] =  anomaly_series[start : end]

        if plot_data:
            self.visualize(stitched_data)

        self.save_data(stitched_data, file_name)
        return stitched_data

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
    
    def save_data(self, data, file_name):
        """Write data to pandas csv file.
        """
        df = pd.DataFrame(data) 
        df.to_csv('data/{}.csv'.format(file_name), header=False, index=False) 

    def load_data(self, file_name='generated_data'):
        """Load data from generated_data.npy file.
        """
        return np.load('data/{file_name}.npy'.format(file_name))

    def visualize(self, data):
        """Plot the generated (stitched) data containing the anomalies.
        """
        fig = plt.figure(1, figsize=(12, 3))
        ax1 = fig.add_subplot(111)
        # Generate title to show window count and anomaly window
        ax1.title.set_text(self.get_title())
        ax1.plot(np.arange(self.nsample), data)
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('img/data.png')
        plt.show()

    def get_title(self):
        """
        Generate title for the data plot containing all anomalies.
        """
        title = 'Generated training data. {} windows stitched together. Window size = {}.'.format(
            self.window_count, 
            self.window_size,
        )
        anomaliesStr = ' Anomalies:'
        for anomaly in self.anomalies:
            start = anomaly * self.window_size
            end = (anomaly + 1) * self.window_size
            anomaliesStr +=  ' ({}:{})'.format(start, end)
        return title + anomaliesStr

    def show_raw_data(self, default_series, anomaly_series):
        """Plot the two time series which will be stitched together.
        """
        fig = plt.figure(1, figsize=(12, 3))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.title.set_text('Normal time series.')
        ax2.title.set_text('Anomaly time series.')
        ax1.plot(np.arange(self.nsample), default_series)
        ax2.plot(np.arange(self.nsample), anomaly_series)
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('img/raw_data.png')
        plt.show()