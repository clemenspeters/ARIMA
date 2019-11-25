import numpy as np
import pandas as pd
import dataGenerator
import dataProcessor
import anomalyDetector
import utils
import matplotlib.pyplot as plt
import terminalColors as tc
plt.style.use('ggplot')


"""
    date                        ,  csv/github,  anomaly-data-index,     anomaly-window-indecies
    "2014-11-01 19:00:00.000000",  5944,        5942,                   118, 117
    "2014-11-27 15:30:00.000000",  7185,        7183,                   143, 142
    "2014-12-25 15:00:00.000000",  8528,        8526,                   170, 169
    "2015-01-01 01:00:00.000000",  8836,        8834,                   176, 175
    "2015-01-27 00:00:00.000000", 10082,        10080,                  201, 200

"""
folder = 'results/taxi_nyc_numenta_data'

def visualize(data, anomalies, title):
    """Plot the generated (stitched) data containing the anomalies.
    """
    fig = plt.figure(1, figsize=(12, 3))
    ax1 = fig.add_subplot(111)

    # Generate title to show window count and anomaly window
    ax1.title.set_text(title)
    ax1.plot(np.arange(data.size), data, color='blue', zorder=1)
    # Add anomaly markers
    for anomaly_index in anomalies:
        ax1.scatter(anomaly_index, data[anomaly_index], marker='x', color='red', zorder=2)
    plt.tight_layout() # avoid overlapping plot titles
    fn = '{}/taxi_data.png'.format(folder)
    fig.savefig(fn)
    tc.green('Created {}'.format(fn))

    # plt.show()


window_size = 100
anomalies = [5942, 7183, 8526, 8834, 10080] # indecies of anomaly timeseries datapoints
# Load timeseries data from csv file
data = pd.read_csv('data/Numenta/data/realKnownCause/nyc_taxi.csv').value.values
visualize(data, anomalies, 'NYC Taxi data')
labels = np.zeros(len(data), dtype=int)
for ai in anomalies:
    labels[ai] = 1


# Generate features
processor = dataProcessor.DataProcessor()
fn = '{}/taxi.csv'.format(folder)
data = processor.generate_features(
    data, 
    labels,
    window_size,
    fn
)
# data = pd.read_csv(fn)
processor.visualize_features(data, fn, method='TSNE')

# Detect anomalies
features = data.drop(['is_anomaly', 'window_label'], axis=1).values
outliers_fraction = data.is_anomaly.mean()
detector = anomalyDetector.AnomalyDetector(
    outliers_fraction, 
    window_size,
    '{}/anomalies.png'.format(folder)
)
detector.detect_anomalies(features)
