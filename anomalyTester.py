import numpy as np
import dataGenerator
import anomalyDetector

def load_data(filename):
    """Load data from file."""
    return np.load(filename)


window_size = 100
window_count = 1000
anomalies = [500, 800]
plot_data = False
generator = dataGenerator.DataGenerator(window_count, window_size, anomalies)

# data = generator.generate_data(plot_data)
data = load_data('generated_data.npy')
# generator.visualize(data)

# Generate features
detector = anomalyDetector.AnomalyDetector(window_size, anomalies)
# features = detector.reduce_arma(data)
features = load_data('features.npy')
detector.visualize_features(features)