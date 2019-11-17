from autoencoder import setup
from autoencoder import encoder
import dataProcessor
import pandas as pd

file_name_training = 'ws100-wc3000-a[500, 1000, 1500, 2000, 2500, 2800]'
file_name_test = 'ws100-wc1000-a[500,800]'
file_name_training_features = '{}--features'.format(file_name_training)
file_name_test_features = '{}--features'.format(file_name_test)

def generate_data_and_features():
    # Generate training data (and features) with anomalies
    setup.generate_data(
        window_size = 100,
        window_count = 3000,
        anomalies = [500, 1000, 1500, 2000, 2500, 2800], # Feature index = * 2 (because stride = window_size/2)
        file_name=file_name_training,
        seed=1111
    )

    # Generate test data (and features) with anomalies
    setup.generate_data(
        window_size = 100,
        window_count = 1000,
        anomalies = [500, 800], # Feature index 1000 and 1600 (because stride = window_size/2)
        file_name=file_name_test,
        seed=55555
    )


def detect_anomalies():
    regularization_strengths = [0.0, 0.0001, 0.001, 0.01, 0.1]

    for regularization_strength in regularization_strengths:

        print('Running with regularization_strength: {}'.format(
            regularization_strength
        ))

        encoder.run(
            file_name_training_features,
            file_name_test_features,
            regularization_strength
        )

def visualize_features(file_name):
    window_size = 100
    anomalies = [500, 1000, 1500, 2000, 2500, 2800]
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    features = pd.read_csv('data/{}.csv'.format(file_name)).values
    processor.visualize_features(features, file_name, 'TSNE')
    processor.visualize_features(features, file_name, 'UMAP')

generate_data_and_features()
visualize_features(file_name_training_features)
detect_anomalies() # Use autoencoder to detect anomalies


# result_file_name = 'autoencoder_anomaly_scores_same_train_test_regularization_dense_0_1'
# encoder.load_and_show(result_file_name, 0.1)