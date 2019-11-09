# Generate data without anomalies (training data)
import dataGenerator
import dataProcessor
import pandas as pd
import matplotlib.pyplot as plt

def generate_training_data(file_name):
    window_size = 100
    window_count = 3000
    anomalies = [] # Feature index 1000 and 1600 (because stride = window_size/2)
    plot_data = False
    file_name_features = '{}--features'.format(file_name)
    generator = dataGenerator.DataGenerator(3000, 100, [])
    data = generator.generate_data(plot_data, file_name)
    # Generate features
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    processor.reduce_arma(data, file_name_features)

def generate_test_data(file_name):
    window_size = 100
    window_count = 1000
    anomalies = [500, 800] # Feature index 1000 and 1600 (because stride = window_size/2)
    plot_data = False
    file_name_features = '{}--features'.format(file_name)
    generator = dataGenerator.DataGenerator(window_count, window_size, anomalies)
    data = generator.generate_data(plot_data, file_name)
    # Generate features
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    processor.reduce_arma(data, file_name_features)


def visualize(file_name):
    data = pd.read_csv('./data/{}.csv'.format(file_name))
    plt.plot(data)
    plt.show()

