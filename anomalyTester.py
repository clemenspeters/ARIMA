import numpy as np
import dataGenerator
import dataProcessor
import anomalyDetector

def load_data(filename):
    """Load data from file."""
    return np.load(filename)


window_size = 100
window_count = 1000
anomalies = [500, 800] # Feature index 1000 and 1600 (because stride = window_size/2)
plot_data = False
generator = dataGenerator.DataGenerator(window_count, window_size, anomalies)

# data = generator.generate_data(plot_data)
# data = load_data('generated_data.npy')
# generator.visualize(data)

# Generate features
# processor = dataProcessor.DataProcessor(window_size, anomalies)
# features = processor.reduce_arma(data)
features = load_data('features.npy')
# processor.visualize_features(features)

detector = anomalyDetector.AnomalyDetector()
detector.detect_anomalies(features)