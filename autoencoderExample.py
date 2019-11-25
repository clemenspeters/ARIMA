from autoencoder import encoder
import dataGenerator
import dataProcessor
import pandas as pd
import terminalColors as tc

folder = 'results/generated/autoencoder'

def generate_training_timeseries():
    # Generate training data (and features) with anomalies
    fn_train = '{}/training_timeseries.csv'.format(folder)

    generator = dataGenerator.DataGenerator(
        window_size = 100,
        anomalies = [500, 1000, 1500, 2000, 2500, 2800],
        window_count = 3000,
        file_name=fn_train,
    )

    # return pd.read_csv(fn_train)
    return generator.generate_timeseries(show = False, seed=1111)


def generate_test_timeseries():
    # Generate test data (and features) with anomalies
    fn_test = '{}/test_timeseries.csv'.format(folder)

    generator = dataGenerator.DataGenerator(
        window_size = 100,
        window_count = 1000,
        anomalies = [500, 800], # Feature index 1000 and 1600 (because stride = window_size/2)
        file_name=fn_test,
    )

    # return pd.read_csv(fn_test)
    return generator.generate_timeseries(show = False, seed=55555)


def generate_features(timeseries, window_size, name='training'):
    encoding_method = 'ARMA'
    fn = '{}/features-{}_{}.csv'.format(folder, name, encoding_method)
    processor = dataProcessor.DataProcessor()

    data = processor.generate_features(
        timeseries.value.values,
        timeseries.is_anomaly.values,
        window_size,
        fn,
        encoding_method
    )

    # data = pd.read_csv(fn)
    fn = '{}/{}-{}'.format(folder, name, encoding_method)
    processor.visualize_features(data, fn, method='TSNE')
    processor.visualize_features(data, fn, method='UMAP')
    return data


def generate_data_and_features():
    test_data = generate_features(ts_test, window_size, 'test')
    window_size = 100
    ts_train = generate_training_timeseries()
    ts_test = generate_test_timeseries()
    train_data = generate_features(ts_train, window_size, 'training')
    test_data = generate_features(ts_test, window_size, 'test')
    return train_data, test_data


def load_data():
    train_data = pd.read_csv('{}/features-training_ARMA.csv'.format(folder))
    test_data = pd.read_csv('{}/features-test_ARMA.csv'.format(folder))
    return train_data, test_data


def detect_anomalies(train_features, test_features, test_labels):
    regularization_strengths = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    regularization_strengths = [0.0001]

    for regularization_strength in regularization_strengths:
        tc.yellow('Running with regularization_strength {}...'.format(
            regularization_strength
        ))

        result_file_name = '{}/anomaly_scores_regularization_{}.csv'.format(
            folder,
            str(regularization_strength).replace('.', '_')
        )

        encoder.run(
            train_features,
            test_features,
            test_labels,
            regularization_strength,
            result_file_name
        )


def visualize_features(file_name):
    window_size = 100
    anomalies = [500, 1000, 1500, 2000, 2500, 2800]
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    features = pd.read_csv('data/{}.csv'.format(file_name)).values
    processor.visualize_features(features, file_name, 'TSNE')
    processor.visualize_features(features, file_name, 'UMAP')

# Generate time series data and ARMA features
# train_data, test_data = generate_data_and_features()
train_data, test_data = load_data()
train_features = train_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_features = test_data.drop(['is_anomaly', 'window_label'], axis=1).values
test_labels = test_data.is_anomaly.values
# Use autoencoder to detect anomalies
detect_anomalies(train_features, test_features, test_labels) 


# result_file_name = 'autoencoder_anomaly_scores_same_train_test_regularization_dense_0_1'
# fn = 'results/generated/autoencoder/anomaly_scores_regularization_0_0001.csv'
# encoder.load_and_show(fn, 0.0001)