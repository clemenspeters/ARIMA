'''
Data source: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Inspired by: https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv
'''

import numpy as np
import pandas as pd
import terminalColors as tc
import matplotlib.pyplot as plt
import dataProcessor
import anomalyDetector
from os import walk

folder = 'aws_lambda_taxi_data/output'
img_dir = 'results/taxi_nyc_custom'

def get_sorted_file_names():
    result = []
    for (dirpath, dirnames, filenames) in walk(folder):
        result.extend(filenames)
        break
    return sorted(result)


def load_files(file_list, file_count):
    '''Load multiple files from a path in correct order.
    '''
    data = []

    for i, fn in enumerate(file_list):
        filename = '{}/{}'.format(folder, fn)
        df = pd.read_csv(filename)
        data.append(df)
        if i >= file_count:
            break

    return pd.concat(data, axis=0, ignore_index=True)


def visualize(data, file_count, file_name, show=False):
    '''Plot a line graph with red markers for all anomalies.
    '''
    fig = plt.figure()
    data.passenger_count.plot.line(color='blue')
    title = 'First {} files nyc taxi (custom) dataset'.format(file_count)
    if file_count == 1:
        title = 'First file nyc taxi (custom) dataset'.format(file_count)
    elif file_count == 126:
        title = title.replace('First', 'All')
    plt.title(title)

    '''
    TODO: Research anomalies (such as Marathon, Blizzard etc.)
    and add an 'is_anomaly' column to the csv files.
    '''
    # Add anomaly markers
    # for index, row in data.loc[data.is_anomaly == 1].iterrows():
    #     plt.scatter(index, row['value'], marker='x', color='red')

    plt.tight_layout()
    fig.savefig(file_name)
    tc.green('Saved plot in {}'.format(file_name))
    if show:
        plt.show()


def print_data_insights(data):
    '''Print some basic information about the data.
    '''
    print(data.head())
    print(data.describe())
    print(data.loc[data.is_anomaly == 1])
    print(data.groupby(['is_anomaly']).mean())
    print(data.groupby(['is_anomaly']).count())
    print(data.index)


file_names = get_sorted_file_names()

# file_count = 10
file_count = len(file_names) # load all files
fn = '{}/plot_all_{}_taxi_files.png'.format(img_dir, file_count)

data = load_files(file_names, file_count)
visualize(data, file_count, fn, show=True)
