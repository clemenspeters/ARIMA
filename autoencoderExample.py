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
encoder.run(file_name_training_features, file_name_test_features)