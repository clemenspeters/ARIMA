import numpy as np
import pandas as pd
import dataGenerator
import dataProcessor
import anomalyDetector
import utils

def generate_timeseries(window_size, window_count, anomaly_windows):
    file_name = 'results/generated/generated_data'
    generator = dataGenerator.DataGenerator(
        window_count,
        window_size,
        anomaly_windows,
        file_name
    )

    timeseries = generator.generate_timeseries(show = False)
    # timeseries = pd.read_csv('results/generated/generated_data.csv')
    # generator.create_data_plot(data, True)
    # generator.visualize(data)
    return timeseries


def generate_features(timeseries, window_size):
    encoding_method = 'ARMA'
    file_name_features = 'results/generated/features_{}.csv'.format(encoding_method)
    processor = dataProcessor.DataProcessor()

    data = processor.generate_features(
        timeseries.value.values,
        timeseries.is_anomaly.values,
        window_size,
        file_name_features,
        encoding_method
    )

    # data = pd.read_csv(file_name_features)
    file_name = 'results/generated/features_{}_TSNE.png'.format(encoding_method)
    processor.visualize_features(data, file_name, method='TSNE')
    file_name = 'results/generated/features_{}_UMAP.png'.format(encoding_method)
    processor.visualize_features(data, file_name, method='UMAP')
    return data


def detect_anomalies(features, outliers_fraction, window_size):
    detector = anomalyDetector.AnomalyDetector(
        outliers_fraction, 
        window_size, 
        'results/generated/anomalies.png'
    )

    detector.detect_anomalies(features)


window_size = 100
timeseries = generate_timeseries(window_size, 1000, [500, 800])
data = generate_features(timeseries, window_size)

features = data.drop(['is_anomaly', 'window_label'], axis=1).values
outliers_fraction = data.is_anomaly.mean()
detect_anomalies(features, outliers_fraction, window_size)