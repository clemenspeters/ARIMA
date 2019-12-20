'''
Data source: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Inspired by: https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv
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
order = (2,2)                 # CHANGE HERE
avoidOverwrite = False           # CHANGE HERE
loadFeatures = False            # CHANGE HERE

folder = 'aws_lambda_taxi_data/output'
result_dir = 'results/taxi_nyc_custom_{}-{}'.format(encoding_method, '_'.join(str(x) for x in order))
window_size = 100

if path.exists('./{}'.format(result_dir)):
    tc.red('RESULT PATH EXISTS ALREADY: {}'.format(result_dir))
    if avoidOverwrite:
        quit()
else:
    os.mkdir(result_dir)

def get_sorted_file_names():
    result = []
    for (dirpath, dirnames, filenames) in walk(folder):
        result.extend(filenames)
        break
    return sorted(result)


def load_files(file_list, file_count):
    '''Load multiple files from a path in correct order.
    '''
    for i, fn in enumerate(file_list):
        filename = '{}/{}'.format(folder, fn)

        if i >= file_count:
            break

        if i == 0:
            series = pd.read_csv(
                filename,
                header=0,
                index_col=0,
                parse_dates=True,
                squeeze=True
            )
            continue

        new_series = pd.read_csv(
            filename,
            header=0,
            index_col=0,
            parse_dates=True,
            squeeze=True
        )

        series = series.append(new_series)

    return series


def visualize(data, file_count, file_name, show=False):
    '''Plot a line graph with red markers for all anomalies.
    '''
    fig = plt.figure()
    data.plot(color='blue')
    title = '{} files of nyc taxi (custom) dataset'.format(file_count)
    plt.title(title)

    '''
    TODO: Research anomalies (such as Marathon, Blizzard etc.)
    and add an 'is_anomaly' column to the csv files.
    '''
    # Add anomaly markers
    # for index, row in data.loc[data.is_anomaly == 1].iterrows():
    #     plt.scatter(index, row['value'], marker='x', color='red')

    plt.tight_layout()
    fig.savefig(file_name)
    tc.green('Saved plot in {}'.format(file_name))
    if show:
        plt.show()


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
        # timeseries.is_anomaly.values,
        np.zeros(len(timeseries.values), dtype=int),
        window_size,
        fn,
        encoding_method,
        order
    )

    # data = pd.read_csv(fn)
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
    file_names = get_sorted_file_names()
    files_test = file_names[30:]
    file_count = len(files_test)
    fn = '{}/plot_test_taxi_files_anomalies.png'.format(result_dir)
    data_test = load_files(files_test, file_count)
    # visualize(data_test, file_count, fn, show=True)

    fig, ax = plt.subplots()
    plt.plot(data_test.index, data_test.values, color='blue')
    title = '{} files nyc taxi (custom) dataset (test data)'.format(file_count)
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
    file_names = get_sorted_file_names()
    files_train = file_names[:30]
    file_count = len(files_train)
    fn = '{}/plot_{}_taxi_files.png'.format(out_folder, file_count)
    data_train = load_files(files_train, file_count)
    visualize(data_train, file_count, fn, show=False)
    train_data = generate_features(data_train, window_size, out_folder, 'train')


    file_names = get_sorted_file_names()
    files_test = file_names[30:]
    file_count = len(files_test)
    fn = '{}/plot_{}_taxi_files.png'.format(out_folder, file_count)
    data_test = load_files(files_test, file_count)
    visualize(data_test, file_count, fn, show=False)
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
scores_fn = '{}/anomaly_scores_regularization_0_01_epochs_100.csv'.format(result_dir)      # CHANGE HERE
threshold = 0.25                                                                           # CHANGE HERE
regularization_strength = 0.01                                                             # CHANGE HERE
features_fn = '{}/features-test_{}.csv'.format(result_dir, encoding_method)
anomaly_windows = label_data(scores_fn, features_fn, regularization_strength, threshold)

# Visualize labelled features
labelled_features_fn = '{}/features-test_{}_labelled_{}.csv'.format(
    result_dir,
    encoding_method,
    str(threshold).replace('.', '_')
)
visualize_labelled_features(labelled_features_fn)

# Show test time series data
visualize_labelled_series(anomaly_windows)
