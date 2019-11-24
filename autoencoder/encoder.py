'''
 # Source: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_14_03_anomaly.ipynb
'''

from sklearn import metrics
import numpy as np
import pandas as pd
# from IPython.display import display, HTML 
import tensorflow as tf
import terminalColors as tc
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


def visualize_and_save(data, file_name, regularization_strength):
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(data) # plotting by columns
    plt.title('regularization_strength: {}'.format(regularization_strength))
    image_file = file_name.replace('.csv', '.png')
    plt.savefig(image_file)
    plt.clf()
    tc.green('Saved image {}'.format(image_file))
    # plt.show()


def run(training_data, test_data, test_anomaly_indices, regularization_strength, file_name):
    assert training_data.shape[1] == test_data.shape[1]

    # Train autoencoder network
    encoding_dim = 2
    model = Sequential()
    data_dim = test_data.shape[1]
    hidden_dim = int(data_dim / 2)
    # Input layer
    model.add(Dense(
        hidden_dim,
        input_dim=data_dim,
        activation='relu',
        activity_regularizer=l1(regularization_strength)
    ))

    # Add layers with decreasing size
    while encoding_dim < hidden_dim:
        model.add(Dense(
            hidden_dim,
            activation='relu',
            activity_regularizer=l1(regularization_strength)
        ))
        hidden_dim = (hidden_dim / 2)

    # Add layers with increasing size
    while hidden_dim < data_dim:
        model.add(Dense(
            hidden_dim,
            activation='relu',
            activity_regularizer=l1(regularization_strength)
        ))
        hidden_dim = (hidden_dim * 2)

    # Output layer
    model.add(Dense(data_dim)) # Multiple output neurons
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(training_data, training_data, verbose=1, epochs=40)

    pred = model.predict(training_data)
    score1 = np.sqrt(metrics.mean_squared_error(pred,training_data))
    print("Training Normal Score (RMSE): {}".format(score1))

    pred = model.predict(test_data)
    score1 = np.sqrt(metrics.mean_squared_error(pred,test_data))
    print("Test Normal Score (RMSE): {}".format(score1))

    # Predict / create anomaly scores
    count = 0
    predictions = []
    for feature in tqdm(test_data):
        pred = model.predict(np.array([feature]))
        score1 = np.sqrt(metrics.mean_squared_error(pred,np.array([feature])))
        predictions.append(score1)
        count += 1
    
    # Save predictions (anomaly scores)
    df = pd.DataFrame(predictions) 
    df.to_csv(file_name, header=False, index=False)
    tc.green('Saved file {}'.format(file_name))

    visualize_and_save(predictions, file_name, regularization_strength)


def load_and_show(file_name, regularization_strength):
    data = pd.read_csv(file_name)
    visualize_and_save(data, file_name, regularization_strength)