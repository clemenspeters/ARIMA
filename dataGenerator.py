# Source: https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_process.html#arma_generate_sample
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(12345)

class DataGenerator:
    def __init__(self, windowCount, windowSize, anomaly):
        self.anomaly = anomaly                  # given windows which should be anomalies
        self.windowSize = windowSize            # number of observations/samples per window
        self.windowCount = windowCount    
        self.sigma = 1                              # standard deviation of noise
        self.distrvs = np.random.randn              # function that generates the random numbers, and takes sample size as argument
        self.nsample = windowCount * windowSize     # number of observations/samples
        self.burnin = 0                             # Burn in observations at the generated and dropped from the beginning of the sample
        self.eta = self.sigma * self.distrvs(self.nsample + self.burnin) # this is where the random samples are drawn. (((Maybe we can insert our anomalies here?)))
        self.defaultSeries = []
        self.anomalySeries = []
        self.stitchedData = []

    def generate_data(self):
        """Stitch together two time series with different  ARMA parameters
        to generate one timeseries which contains anomalies.
        """
        # # Genrate the two timeseries (with different ARMA parameters)
        self.defaultSeries = self.arma_generate_sample([.75, -.25], [.65, .35])
        self.anomalySeries = self.arma_generate_sample([.75, -.25], [-.65, .35])
        # Plot the two timeseries
        self.show_raw_data()
        # Combine the two timeseries to get one time series containing anomalies
        anomalyStart = self.anomaly[0]
        anomalyEnd = self.anomaly[1]
        self.stitchedData = np.hstack((
            self.defaultSeries[self.burnin + 0 : anomalyStart + self.burnin],
            self.anomalySeries[self.burnin + anomalyStart : anomalyEnd + self.burnin],
            self.defaultSeries[self.burnin + anomalyEnd :],
        ))
        return self.stitchedData

    def arma_generate_sample(self, ar, ma):
        """
        See: https://www.statsmodels.org/devel/_modules/statsmodels/tsa/arima_process.html#ArmaProcess.generate_sample
        """
        arparams = np.array(ar)
        maparams = np.array(ma)
        arparams = np.r_[1, -arparams] # add zero-lag and negate
        maparams = np.r_[1, maparams]  # add zero-lag
        return signal.lfilter(maparams, arparams, self.eta, axis=0)
    
    def visualize(self):
        """Plot the generated (stitched) data containing the anomalies.
        """
        fig = plt.figure(1, figsize=(12, 3))
        ax1 = fig.add_subplot(111)
        # Generate title to show window count and anomaly window
        anomalyStart = self.anomaly[0]
        anomalyEnd = self.anomaly[1]
        ax1.title.set_text('Generated training data. {} windows stitched together. Window size = {}. Anomaly: ({}:{})'.format(
            int(self.nsample / self.windowSize), 
            self.windowSize,
            anomalyStart,
            anomalyEnd
        ))
        ax1.plot(np.arange(self.nsample), self.stitchedData[self.burnin:])
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('data.png')
        plt.show()


    def show_raw_data(self):
        """Plot the two time series which will be stitched together.
        """
        fig = plt.figure(1, figsize=(12, 3))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.title.set_text('Normal time series.')
        ax2.title.set_text('Anomaly time series.')
        ax1.plot(np.arange(self.nsample), self.defaultSeries)
        ax2.plot(np.arange(self.nsample), self.anomalySeries)
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('raw_data.png')
        plt.show()