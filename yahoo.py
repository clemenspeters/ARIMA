# Data source: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1
# yahoo-data-labeled-time-series-anomalies-v1_0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir = './data/yahoo-data-labeled-time-series-anomalies-v1_0/A1Benchmark'
file_numers = np.arange(1, 67)
# Load the data from the csv file
data = pd.read_csv('{}/real_{}.csv'.format(dir, file_numers[0]))

# Print some insights on the data
print(data.head())
print(data.describe())
print(data.loc[data.is_anomaly == 1])

# Visualize / plot
data.is_anomaly.plot.line()
data.value.plot.line()
plt.show()