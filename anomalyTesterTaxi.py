import numpy as np
import pandas as pd
import dataGenerator
import dataProcessor
import anomalyDetector

import matplotlib.pyplot as plt
plt.style.use('ggplot')


"""
    date                        ,  csv/github,  data-index, window-inde
    "2014-11-01 19:00:00.000000",  5944,        5942,       118, 117 // 
    "2014-11-27 15:30:00.000000",  7185,        7183,       143, 142
    "2014-12-25 15:00:00.000000",  8528,        8526,       170, 169
    "2015-01-01 01:00:00.000000",  8836,        8834,       176, 175 //
    "2015-01-27 00:00:00.000000", 10082,        10080,      201, 200

"""


def visualize(data, title):
    """Plot the generated (stitched) data containing the anomalies.
    """
    fig = plt.figure(1, figsize=(12, 3))
    ax1 = fig.add_subplot(111)
    # Generate title to show window count and anomaly window
    ax1.title.set_text(title)
    ax1.plot(np.arange(data.size), data)
    plt.tight_layout() # avoid overlapping plot titles
    fig.savefig('results/numenta/taxi_data.png')
    plt.show()


window_size = 100
# anomalies = [118, 143, 170, 176, 201] # indecies of windows
anomalies = [117,118, 142,143, 169,170, 175,176, 200,201] # indecies of windows

csv_data = pd.read_csv('data/Numenta/data/realKnownCause/nyc_taxi.csv')
data = np.array(csv_data[['value']].get_values())[:, 0]
print('number of data points:', data.shape[0])
# visualize(data, 'NYC Taxi data')

# Generate features
processor = dataProcessor.DataProcessor(window_size, anomalies, 'results/numenta/taxi_features')
# features = processor.reduce_arma(data)
features = processor.load_data()
processor.visualize_features(features)
print('number of features:', features.shape[0])

outliers_fraction = 10 / 205
detector = anomalyDetector.AnomalyDetector(
    outliers_fraction, 
    window_size,
    'results/numenta/anomalies'
)
detector.detect_anomalies(features)