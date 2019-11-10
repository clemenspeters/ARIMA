# Data source: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1
# yahoo-data-labeled-time-series-anomalies-v1_0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_files(path, file_count):
    '''Load multiple files from a path in correct order.
    '''
    file_numers = np.arange(1, file_count + 1)
    file_list = []

    for number in file_numers:
        filename = '{}/real_{}.csv'.format(path, number)
        df = pd.read_csv(filename, index_col=None, header=0)
        file_list.append(df)

    return pd.concat(file_list, axis=0, ignore_index=True)

def visualize(data):
    '''Plot a line graph with red markers for all anomalies.
    '''
    fig = plt.figure()
    data.value.plot.line()

    # Add anomaly markers
    for index, row in data.loc[data.is_anomaly == 1].iterrows():
        plt.scatter(index, row['value'], marker='x', color='red')

    plt.show()


def plot_data_insights(data):
    '''Print some basic information about the data.
    '''
    print(data.head())
    print(data.describe())
    print(data.loc[data.is_anomaly == 1])
    print(data.groupby(['is_anomaly']).mean())
    print(data.groupby(['is_anomaly']).count())
    print(data.index)

path = './data/yahoo-data-labeled-time-series-anomalies-v1_0/A1Benchmark'
data = load_files(path, 67) # Load the combined data from the 67 csv files
plot_data_insights(data) # Print some insights on the data
visualize(data) # Visualize / plot

