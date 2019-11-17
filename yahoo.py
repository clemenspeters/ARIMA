# Data source: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1
# yahoo-data-labeled-time-series-anomalies-v1_0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataProcessor
import anomalyDetector
import utils


def load_files(path, file_count):
    '''Load multiple files from a path in correct order.
    '''
    file_numers = np.arange(1, file_count + 1)
    file_list = []

    for number in file_numers:
        filename = '{}/real_{}.csv'.format(path, number)
        df = pd.read_csv(filename, index_col=None, header=0)
        file_list.append(df)

    return pd.concat(file_list, axis=0, ignore_index=True)


def visualize(data, file_count):
    '''Plot a line graph with red markers for all anomalies.
    '''
    fig = plt.figure()
    data.value.plot.line(color='blue')
    title = 'First {} files of yahoo S5 dataset'.format(file_count)
    if file_count == 1:
        title = 'First file of yahoo S5 dataset'.format(file_count)
    plt.title(title)

    # Add anomaly markers
    for index, row in data.loc[data.is_anomaly == 1].iterrows():
        plt.scatter(index, row['value'], marker='x', color='red')

    plt.tight_layout()
    fig.savefig('results/yahoo_data_{}_files.png'.format(file_count))
    plt.show()


def plot_data_insights(data):
    '''Print some basic information about the data.
    '''
    print(data.head())
    print(data.describe())
    print(data.loc[data.is_anomaly == 1])
    print(data.groupby(['is_anomaly']).mean())
    print(data.groupby(['is_anomaly']).count())
    print(data.index)


# Load data
path = './data/yahoo-data-labeled-time-series-anomalies-v1_0/A1Benchmark'
file_count = 1 # <= 67
data = load_files(path, file_count) # Load the combined data from the 67 csv files
plot_data_insights(data) # Print some insights on the data
visualize(data, file_count) # Visualize / plot

# Get anomaly labels from data
window_size = 100
anomalies = data.index[data['is_anomaly'] == 1].tolist()
anomaly_windows = utils.anomalies_index_to_window_index(anomalies, window_size)

# Generate features
features_file_name = 'yahoo-features-{}'.format(file_count)
processor = dataProcessor.DataProcessor(window_size, anomaly_windows)
features = processor.reduce_arma(data.value, features_file_name)
features = pd.read_csv('data/{}.csv'.format(features_file_name)).values
processor.visualize_features(features, features_file_name)

# Detect anomalies
outliers_fraction = 2 / 27
detector = anomalyDetector.AnomalyDetector(
    outliers_fraction, 
    window_size, 
    'img/anomalies'
)
detector.detect_anomalies(features)


