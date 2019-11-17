# Generate data without anomalies (training data)
import dataGenerator
import dataProcessor
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(window_size, window_count, anomalies, file_name, seed):
    plot_data = False
    file_name_features = '{}--features'.format(file_name)
    generator = dataGenerator.DataGenerator(window_count, window_size, anomalies)
    data = generator.generate_data(plot_data, file_name, seed)
    # Generate features
    processor = dataProcessor.DataProcessor(window_size, anomalies)
    processor.reduce_arma(data, file_name_features)


def visualize(file_name):
    data = pd.read_csv('./data/{}.csv'.format(file_name))
    plt.plot(data)
    plt.show()

