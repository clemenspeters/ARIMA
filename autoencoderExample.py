from autoencoder import setup
from autoencoder import encoder

file_name_training = 'ws100-wc3000-a[]'
file_name_test = 'ws100-wc1000-a[500,800]'
file_name_training_features = '{}--features'.format(file_name_training)
file_name_test_features = '{}--features'.format(file_name_test)

# Gennerate data and features
setup.generate_training_data(file_name_training)
setup.generate_test_data(file_name_test)

# Use autoencoder to detect anomalies
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



# result_file_name = 'autoencoder_anomaly_scores_same_train_test_regularization_dense_0_1'
# encoder.load_and_show(result_file_name, 0.1)