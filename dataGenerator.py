# Source: https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_process.html#arma_generate_sample
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(12345)

class DataGenerator:
    def __init__(self):
        # First set of ARMA parameters
        arparams = np.array([.75, -.25])
        maparams = np.array([.65, .35])
        self.arparams = np.r_[1, -arparams] # add zero-lag and negate
        self.maparams = np.r_[1, maparams]  # add zero-lag

        # Second set of ARMA parameters
        arparams2 = np.array([.75, .25])
        maparams2 = np.array([-.65, .35])
        self.arparams2 = np.r_[1, -arparams2] # add zero-lag and negate
        self.maparams2 = np.r_[1, maparams2]  # add zero-lag

        self.sigma = 1                   # standard deviation of noise
        self.distrvs = np.random.randn   # function that generates the random numbers, and takes sample size as argument
        self.nsample = 500               # number of observations/samples
        self.burnin = 0                  # Burn in observations at the generated and dropped from the beginning of the sample
        self.eta = self.sigma * self.distrvs(self.nsample + self.burnin) # this is where the random samples are drawn. (((Maybe we can insert our anomalies here?)))
        self.y = []

    def getData(self):
        # Now we stich together sections with different ARMA parameters
        self.y = np.hstack((
            signal.lfilter(self.maparams, self.arparams, self.eta[0:50 + self.burnin]), 
            signal.lfilter(self.maparams, self.arparams, self.eta[self.burnin + 50:100 + self.burnin]), 
            signal.lfilter(self.maparams2, self.arparams2, self.eta[self.burnin + 100:150 + self.burnin]), 
            signal.lfilter(self.maparams, self.arparams, self.eta[self.burnin + 150:]), 
        ))
        return self.y
    
    def visualize(self):
        fig = plt.figure(1, figsize=(12, 3))
        ax4 = fig.add_subplot(111)
        ax4.title.set_text("Generated training data. Four windows stitched together. Window size = 50")
        ax4.plot(np.arange(self.nsample), self.y[self.burnin:])
        plt.tight_layout() # avoid overlapping plot titles
        fig.savefig('data.png')
        plt.show()