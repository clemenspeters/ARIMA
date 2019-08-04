# Source: https://www.statsmodels.org/stable/_modules/statsmodels/tsa/arima_process.html#arma_generate_sample
from scipy import signal
import numpy as np
# import matplotlib.style
import matplotlib.pyplot as plt
plt.style.use('ggplot')

np.random.seed(12345)

# First set of ARMA parameters
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams] # add zero-lag and negate
maparams = np.r_[1, maparams]  # add zero-lag

# Second set of ARMA parameters
arparams2 = np.array([.75, .25])
maparams2 = np.array([-.65, .35])
arparams2 = np.r_[1, -arparams2] # add zero-lag and negate
maparams2 = np.r_[1, maparams2]  # add zero-lag

# print('arparams', arparams)
# print('maparams', maparams)

sigma = 1                   # standard deviation of noise
distrvs = np.random.randn   # function that generates the random numbers, and takes sample size as argument
nsample = 200               # number of observations/samples
burnin = 0                  # Burn in observations at the generated and dropped from the beginning of the sample
eta = sigma * distrvs(nsample + burnin) # this is where the random samples are drawn. (((Maybe we can insert our anomalies here?)))
# y = signal.lfilter(maparams, arparams, eta)

# Now we stich together sections with different ARMA parameters
y = np.hstack((
    signal.lfilter(maparams, arparams, eta[0:50 + burnin]), 
    signal.lfilter(maparams, arparams, eta[burnin + 50:100 + burnin]), 
    signal.lfilter(maparams2, arparams2, eta[burnin + 100:150 + burnin]), 
    signal.lfilter(maparams, arparams, eta[burnin + 150:]), 
#     eta[100:]
))

y1 = signal.lfilter(maparams, arparams, eta)[burnin:]
y2 = signal.lfilter(maparams2, arparams2, eta)[burnin:]

fig = plt.figure(1, figsize=(12, 6))
# ax1 = fig.add_subplot(411)
# ax1.title.set_text("Drawn from normal distribution")
# ax1.plot(np.arange(nsample), eta[burnin:])

ax2 = fig.add_subplot(412)
ax2.title.set_text("ARMA Parameter set 1")
ax2.plot(np.arange(nsample), y1)

ax3 = fig.add_subplot(413)
ax3.title.set_text("ARMA Parameter set 2")
ax3.plot(np.arange(nsample), y2)

ax4 = fig.add_subplot(414)
ax4.title.set_text("Generated training data. Four windows stitched together. Window size = 50")
ax4.plot(np.arange(nsample), y[burnin:])


plt.tight_layout() # avoid overlapping plot titles
fig.savefig('test.png')