import numpy as np
import dataGenerator
import anomalyDetector

def load_data(filename):
    """Load data from file."""
    return np.load(filename)

def generate_features(data, window_size, anomalies):
    # Generate features
    detector = anomalyDetector.AnomalyDetector(window_size, anomalies)
    detector.reduce_arma(data)
    detector.visualize_features()

window_size = 100
window_count = 1000
anomalies = [500]
plot_data = True
generator = dataGenerator.DataGenerator(window_count, window_size, anomalies)

# data = generator.generate_data(plot_data)
data = load_data('generated_data.npy')
# generator.visualize(data)

generate_features(data, window_size, anomalies)