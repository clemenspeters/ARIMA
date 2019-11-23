import numpy as np
import pandas as pd
import dataGenerator
import dataProcessor
import anomalyDetector
import utils
import matplotlib.pyplot as plt
plt.style.use('ggplot')


"""
    date                        ,  csv/github,  anomaly-data-index,     anomaly-window-indecies
    "2014-11-01 19:00:00.000000",  5944,        5942,                   118, 117
    "2014-11-27 15:30:00.000000",  7185,        7183,                   143, 142
    "2014-12-25 15:00:00.000000",  8528,        8526,                   170, 169
    "2015-01-01 01:00:00.000000",  8836,        8834,                   176, 175
    "2015-01-27 00:00:00.000000", 10082,        10080,                  201, 200

"""

def visualize(data, anomalies, title):
    """Plot the generated (stitched) data containing the anomalies.
    """
    fig = plt.figure(1, figsize=(12, 3))
    ax1 = fig.add_subplot(111)

    # Generate title to show window count and anomaly window
    ax1.title.set_text(title)
    ax1.plot(np.arange(data.size), data, color='blue', zorder=-1)
    # Add anomaly markers
    for anomaly_index in anomalies:
        ax1.scatter(anomaly_index, data[anomaly_index], marker='x', color='red', zorder=1)
    plt.tight_layout() # avoid overlapping plot titles
    fig.savefig('taxi_nyc_numenta_data/taxi_data.png')
    # plt.show()


window_size = 100
anomalies = [5942, 7183, 8526, 8834, 10080] # indecies of anomaly timeseries datapoints
# Load timeseries data from csv file
data = pd.read_csv('data/Numenta/data/realKnownCause/nyc_taxi.csv').values[:, 1]
visualize(data, anomalies, 'NYC Taxi data')

# anomaly_windows = [117,118, 142,143, 169,170, 175,176, 200,201] # two windows per anomaly
anomaly_windows = utils.anomalies_index_to_window_index(anomalies, window_size)
print(anomaly_windows)

# Generate features
processor = dataProcessor.DataProcessor(window_size, anomaly_windows)
features_file_name = 'taxi_nyc_numenta_data/taxi_features'
features = processor.reduce_arma(data, features_file_name)
features = pd.read_csv('{}.csv'.format(features_file_name)).values
processor.visualize_features(features, features_file_name)

# Detect anomalies
outliers_fraction = 10 / 205
detector = anomalyDetector.AnomalyDetector(
    outliers_fraction, 
    window_size,
    'taxi_nyc_numenta_data/anomalies'
)
detector.detect_anomalies(features)
