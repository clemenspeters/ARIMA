'''
Data source: https://physionet.org/content/qtdb/1.0.0/
'''

import numpy as np
import pandas as pd
import terminalColors as tc
import matplotlib.pyplot as plt
import matplotlib as mpl
import dataProcessor
import anomalyDetector
from os import walk
import os
from os import path
from autoencoder import encoder
import terminalColors as tc

# encoding_method = 'ARIMA'       # CHANGE HERE
# order = (3, 1, 3)               # CHANGE HERE
encoding_method = 'ARMA'      # CHANGE HERE
order = (2, 2)                 # CHANGE HERE
avoidOverwrite = False           # CHANGE HERE
loadFeatures = False            # CHANGE HERE

folder = 'data/ecg'
result_dir = 'results/ecg'
window_size = 400



def print_data_insights(data):
    '''Print some basic information about the data.
    '''
    print(data.head())
    print(data.describe())
    print(data.loc[data.is_anomaly == 1])
    print(data.groupby(['is_anomaly']).mean())
    print(data.groupby(['is_anomaly']).count())
    print(data.index)


def generate_features(timeseries, window_size, out_folder, name='training'):
    fn = '{}/features-{}_{}.csv'.format(out_folder, name, encoding_method)
    processor = dataProcessor.DataProcessor()

    data = processor.generate_features(
        timeseries.values,
        np.zeros(len(timeseries.values), dtype=int),
        window_size,
        fn,
        encoding_method,
        order
    )

    fn = '{}/{}-{}'.format(out_folder, name, encoding_method)
    processor.visualize_features(data, fn, method='TSNE')
    processor.visualize_features(data, fn, method='UMAP')
    return data

def visualize_labelled_features(file_name, show=True):
    features = pd.read_csv(file_name)
    processor = dataProcessor.DataProcessor()
    processor.visualize_features(features, file_name, 'TSNE', show)
    processor.visualize_features(features, file_name, 'UMAP', show)

def visualize_labelled_series(anomaly_windows, show=True):
    fn = '{}/plot_anomalies.png'.format(result_dir)
    # data_test = load_files(files_test, file_count)
    fn = '{}/sel102.csv'.format(result_dir)
    df = pd.read_csv(fn, header=0)
    row_count = len(df.index)
    split = int(row_count / 2)
    # data_train = df.iloc[split:].V5
    data_test = df.iloc[:-split].V5

    fig, ax = plt.subplots()
    plt.plot(data_test.index, data_test.values, color='blue')
    title = 'ECG dataset (test data)'
    plt.title(title)

    # Highlight anomalies
    for window in anomaly_windows:
        start, end  = window.split('-')
        start = int(start)
        end = int(end)
        if len(data_test) < end:
            end = len(data_test)
        plt.plot(data_test.index[start:end], data_test.values[start:end], color='red')

    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()
    fn = fn.replace('.csv', '.png')
    fig.savefig(fn)
    tc.green('Saved plot in {}'.format(fn))
    if show:
        plt.show()

def detect_anomalies(train_features, test_features, test_labels, out_folder):
    regularization_strengths = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    epochs = 100

    for regularization_strength in regularization_strengths:
        tc.yellow('Running with regularization_strength {}...'.format(
            regularization_strength
        ))

        fn = '{}/anomaly_scores_regularization_{}_epochs_{}.csv'.format(
            out_folder,
            str(regularization_strength).replace('.', '_'),
            epochs
        )

        encoder.run(
            train_features,
            test_features,
            test_labels,
            regularization_strength,
            fn,
            epochs
        )


def generate_data_and_features(out_folder):
    fn = '{}/sel102.csv'.format(result_dir)
    df = pd.read_csv(fn, header=0)
    row_count = len(df.index)
    split = int(row_count / 2)
    data_train = df.iloc[split:].V5
    data_test = df.iloc[:-split].V5

    train_data = generate_features(data_train, window_size, out_folder, 'train')
    test_data = generate_features(data_test, window_size, out_folder, 'test')

    return train_data, test_data


def load_data():
    test_data = pd.read_csv('{}/features-test_{}.csv'.format(
        result_dir, 
        encoding_method
    ))
    train_data = pd.read_csv('{}/features-train_{}.csv'.format(
        result_dir, 
        encoding_method
    ))
    return train_data, test_data

def label_data(scores_fn, features_fn, regularization_strength, threshold):
    encoder.load_and_show(scores_fn, regularization_strength)
    return encoder.load_and_label_data(features_fn, threshold, scores_fn)

def show_labelled_data(labelled_features_fn):
    labelled_features = pd.read_csv(labelled_features_fn)
    mask = labelled_features.is_anomaly > 0
    anomaly_windows = anomaly_windows = labelled_features.loc[mask].window_label
    visualize_labelled_series(anomaly_windows)

def show_raw_data():
    file_names = get_sorted_file_names()
    selected_files = file_names[:30]
    file_count = len(selected_files)
    fn = '{}/plot_all_{}_taxi_files.png'.format('test', file_count)
    data_train = load_files(selected_files, file_count)
    visualize(data_train, file_count, fn, show=True)


if loadFeatures:
    train_data, test_data = load_data()
else:
    train_data, test_data = generate_data_and_features(result_dir)

train_features = train_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_features = test_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_labels = test_data.is_anomaly.values
# Use autoencoder to detect anomalies
detect_anomalies(train_features, test_features, test_labels, result_dir) 

# Set anomaly labels on given threshold of anomaly scores
scores_fn = '{}/anomaly_scores_regularization_0_001_epochs_100.csv'.format(result_dir)      # CHANGE HERE
threshold = 0.18                                                                           # CHANGE HERE
regularization_strength = 0.001                                                             # CHANGE HERE
features_fn = '{}/features-test_{}.csv'.format(result_dir, encoding_method)
anomaly_windows = label_data(scores_fn, features_fn, regularization_strength, threshold)

# # Visualize labelled features
labelled_features_fn = '{}/features-test_{}_labelled_{}.csv'.format(
    result_dir,
    encoding_method,
    str(threshold).replace('.', '_')
)
visualize_labelled_features(labelled_features_fn)

# # Show test time series data
visualize_labelled_series(anomaly_windows)


# Use already labelled feature file to show anomalies in time series data
# labelled_features_fn = 'results/taxi_nyc_custom_ARMA-2_2/features-test_ARMA_labelled_0_25.csv'
# show_labelled_data(labelled_features_fn)
