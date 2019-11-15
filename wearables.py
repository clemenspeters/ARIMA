'''
Data source: http://cogentee.coventry.ac.uk/datasets/fall_adl_data.zip
Related reserach paper: https://www.researchgate.net/publication/224991506_Recognition_of_Human_Motion_Related_Activities_from_Sensors
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataProcessor
import anomalyDetector


def visualize(data, column, title):
    '''Plot a line graph with red markers for all anomalies.
    '''
    plt.title(title)
    data[column].plot.line()
    plt.show()


def plot_data_insights(data):
    '''Print some basic information about the data.
    '''
    print(data.head())
    print(data.describe())


# Load 'falls' data
filename = './data/wearables/falls/subject_1'
falls = pd.read_csv(filename, index_col=None, header=0)

# Load 'walking on stairs' data
filename = './data/wearables/walking on stairs/subject33'
stairs = pd.read_csv(filename, index_col=None, header=0)

# "falls" and "walking on stairs" have the same column names
columns = [
    "ch_accel_x",
    "ch_accel_y",
    # "ch_accel_z",
    # "ch_gyro_x",
    # "ch_gyro_y",
    # "ch_gyro_z",
    # "th_accel_x",
    # "th_accel_y",
    # "th_accel_z",
    # "th_gyro_x",
    # "th_gyro_y",
    # "th_gyro_z"
]

# Vizualize 'falls' data
for column in columns:
    plot_data_insights(falls)
    visualize(falls, column, 'Falls: {}'.format(column))

# Vizualize 'walking on stairs' data
for column in columns:
    plot_data_insights(falls)
    visualize(falls, column, 'Stairs: {}'.format(column))

