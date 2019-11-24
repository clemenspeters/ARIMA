# Data source: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1
# yahoo-data-labeled-time-series-anomalies-v1_0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataProcessor
import anomalyDetector
import utils
import terminalColors as tc


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


def visualize(data, file_count, file_name, show=False):
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
    fig.savefig(file_name)
    tc.green('Saved plot in {}'.format(file_name))
    if show:
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

folder = 'results/yahoo'
# Load data
path = './data/yahoo-data-labeled-time-series-anomalies-v1_0/A1Benchmark'
file_count = 1 # <= 67
data = load_files(path, file_count) # Load the combined data from the 67 csv files
# plot_data_insights(data) # Print some insights on the data
fn = '{}/yahoo_data_{}_files.png'.format(folder, file_count)
visualize(data, file_count, fn) # Visualize / plot

# Get anomaly labels from data
window_size = 100
anomalies = data.index[data['is_anomaly'] == 1].tolist()
anomaly_windows = utils.anomalies_index_to_window_index(anomalies, window_size)

# Generate features
encoding_method = 'ARMA'
fn = '{}/yahoo-features-{}_{}.csv'.format(folder, file_count, encoding_method)
processor = dataProcessor.DataProcessor()
features = processor.generate_features(
    data.value.values, 
    data.is_anomaly.values,
    window_size,
    fn,
    encoding_method
)
# features = pd.read_csv(fn)
fn = '{}/features_{}_TSNE.png'.format(folder, encoding_method)
processor.visualize_features(features, fn, method='TSNE')

# Detect anomalies
outliers_fraction = features.is_anomaly.mean()
fn = '{}/anomalies.png'.format(folder)

detector = anomalyDetector.AnomalyDetector(
    outliers_fraction, 
    window_size, 
    fn
)

features = features.drop(['is_anomaly', 'window_label'], axis=1).values
detector.detect_anomalies(features)


