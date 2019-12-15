'''
Data source: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Inspired by: https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv
'''

import numpy as np
import pandas as pd
import terminalColors as tc
import matplotlib.pyplot as plt
import dataProcessor
import anomalyDetector
from os import walk
import os
from os import path
from autoencoder import encoder
import terminalColors as tc

encoding_method = 'ARIMA'       # CHANGE HERE
order = (3, 1, 3)               # CHANGE HERE
# encoding_method = 'ARMA'      # CHANGE HERE
# order = (2,2)                 # CHANGE HERE
avoidOverwrite = True           # CHANGE HERE

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
    title = 'First {} files nyc taxi (custom) dataset'.format(file_count)
    if file_count == 1:
        title = 'First file nyc taxi (custom) dataset'.format(file_count)
    elif file_count == 126:
        title = title.replace('First', 'All')
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
    encoding_method = 'ARMA'
    fn = '{}/features-{}_{}.csv'.format(out_folder, name, encoding_method)
    processor = dataProcessor.DataProcessor()

    data = processor.generate_features(
        timeseries.values,
        # timeseries.is_anomaly.values,
        np.zeros(len(timeseries.values), dtype=int),
        window_size,
        fn,
        encoding_method
    )

    # data = pd.read_csv(fn)
    fn = '{}/{}-{}'.format(out_folder, name, encoding_method)
    processor.visualize_features(data, fn, method='TSNE')
    processor.visualize_features(data, fn, method='UMAP')
    return data


def visualize_features(file_name, window_size):
    anomalies = []
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    features = pd.read_csv('data/{}.csv'.format(file_name)).values
    processor.visualize_features(features, file_name, 'TSNE')
    processor.visualize_features(features, file_name, 'UMAP')

def visualize_labelled_features(file_name, show=True):
    features = pd.read_csv(file_name)
    processor = dataProcessor.DataProcessor()
    processor.visualize_features(features, file_name, 'TSNE', show)

def visualize_labelled_series(anomaly_windows, show=True):
    file_names = get_sorted_file_names()
    files_test = file_names[30:]
    file_count = len(files_test)
    fn = '{}/plot_all_{}_taxi_files.png'.format(result_dir, file_count)
    data_test = load_files(files_test, file_count)
    # visualize(data_test, file_count, fn, show=True)

    fig = plt.figure()
    plt.plot(data_test.values, color='blue')
    title = '{} files nyc taxi (custom) dataset (test data)'.format(file_count)
    plt.title(title)

    # Highlight anomalies
    for window in anomaly_windows:
        start, end  = window.split('-')
        start = int(start)
        end = int(end)
        if len(data_test) < end:
            end = len(data_test)
        plt.plot(range(start, end), data_test.values[start:end], color='red')

    plt.tight_layout()
    # fig.savefig(file_name)
    # tc.green('Saved plot in {}'.format(file_name))
    if show:
        plt.show()

def detect_anomalies(train_features, test_features, test_labels, out_folder):
    regularization_strengths = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    for regularization_strength in regularization_strengths:
        tc.yellow('Running with regularization_strength {}...'.format(
            regularization_strength
        ))

        result_file_name = '{}/anomaly_scores_regularization_{}.csv'.format(
            out_folder,
            str(regularization_strength).replace('.', '_')
        )

        encoder.run(
            train_features,
            test_features,
            test_labels,
            regularization_strength,
            result_file_name
        )


def generate_data_and_features(out_folder):
    file_names = get_sorted_file_names()
    files_train = file_names[:30]
    file_count = len(file_names) # load all files
    fn = '{}/plot_all_{}_taxi_files.png'.format(out_folder, file_count)
    data_train = load_files(files_train, file_count)
    visualize(data_train, file_count, fn, show=False)
    train_data = generate_features(data_train, window_size, out_folder, 'train')


    file_names = get_sorted_file_names()
    files_test = file_names[30:]
    file_count = len(file_names) # load all files
    fn = '{}/plot_all_{}_taxi_files.png'.format(out_folder, file_count)
    data_test = load_files(files_test, file_count)
    visualize(data_test, file_count, fn, show=False)
    test_data = generate_features(data_test, window_size, out_folder, 'test')

    return train_data, test_data


def load_data():
    test_data = pd.read_csv('{}/features-test_ARMA.csv'.format(result_dir))
    train_data = pd.read_csv('{}/features-train_ARMA.csv'.format(result_dir))
    return train_data, test_data

def label_data(regularization_strength, threshold):
    scores_fn = '{}/anomaly_scores_regularization_{}.csv'.format(
        result_dir,
        str(regularization_strength).replace('.', '_')
    )
    features_fn = '{}/features-test_ARMA.csv'.format(result_dir)
    encoder.load_and_show(scores_fn, regularization_strength)
    return encoder.load_and_label_data(features_fn, threshold, scores_fn)

# train_data, test_data = generate_data_and_features(result_dir)

train_data, test_data = load_data()
train_features = train_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_features = test_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_labels = test_data.is_anomaly.values
# Use autoencoder to detect anomalies
detect_anomalies(train_features, test_features, test_labels, result_dir) 

# Set anomaly labels on given threshold of anomaly scores
anomaly_windows = label_data(0.001, 0.2)

# Visualize labelled features
labelled_features_fn = '{}/features-test_ARMA_labelled_0.2.csv'.format(result_dir)
visualize_labelled_features(labelled_features_fn)

# Show test time series data
visualize_labelled_series(anomaly_windows)