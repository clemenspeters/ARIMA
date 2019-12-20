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
import os
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import plot_model


def visualize_and_save(
    data,
    labels,
    file_name,
    regularization_strength,
    show=False
):
    # Show green vertical lines for each anomaly label
    # anomaly_indices = [i for i, x in enumerate(labels) if x == 1]
    # for ai in anomaly_indices:
    #     plt.axvline(x=ai, zorder=-1, c='green')
    
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(data) # plotting by columns
    plt.title('regularization_strength: {}'.format(regularization_strength))
    image_file = file_name.replace('.csv', '.png')
    plt.savefig(image_file)
    tc.green('Saved image {}'.format(image_file))
    if show:
        plt.show()
    plt.clf()


def run(
    training_data,
    test_data,
    test_labels,
    regularization_strength,
    file_name,
    epochs=100
):
    assert training_data.shape[1] == test_data.shape[1]

    # Train autoencoder network
    encoding_dim = 2
    model = Sequential()
    data_dim = test_data.shape[1]
    layers = [data_dim]
    hidden_dim = int(data_dim / 2)
    # Input layer and first encoding layer
    model.add(Dense(
        hidden_dim,
        input_dim=data_dim,
        activation='relu',
        activity_regularizer=l1(regularization_strength),
        name='encoding_{}'.format(hidden_dim)
    ))
    layers.append(hidden_dim)

    # Add layers with decreasing size
    hidden_dim = int(hidden_dim / 2)
    while encoding_dim <= hidden_dim:
        model.add(Dense(
            hidden_dim,
            activation='relu',
            activity_regularizer=l1(regularization_strength),
            name='encoding_{}'.format(hidden_dim)
        ))
        layers.append(hidden_dim)
        hidden_dim = int(hidden_dim / 2)
    
    # Add layers with increasing size
    layers.pop() # remove smallest element
    for hidden_dim in sorted(layers):
        model.add(Dense(
            hidden_dim,
            activation='relu',
            activity_regularizer=l1(regularization_strength),
            name='decoding_{}'.format(hidden_dim)
        ))

    # Output layer
    model.add(Dense(data_dim, name='output')) # Multiple output neurons
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(training_data, training_data, verbose=1, epochs=epochs)

    # Save network structure to png
    dirname = os.path.dirname(file_name)
    fn = '{}/auto_encoder_model.png'.format(dirname)
    plot_model(model, to_file=fn, show_shapes=True)
    tc.green('Saved model image as {}'.format(fn))

    pred = model.predict(training_data)
    score = np.sqrt(metrics.mean_squared_error(pred, training_data))
    tc.yellow("Training Normal Score (RMSE): {}".format(score))

    pred = model.predict(test_data)
    score = np.sqrt(metrics.mean_squared_error(pred, test_data))
    tc.yellow("Test Normal Score (RMSE): {}".format(score))

    # Predict / create anomaly scores
    scores = []
    tc.yellow('Generating anomaly scores...')
    for feature in tqdm(test_data):
        pred = model.predict(np.array([feature]))
        score = np.sqrt(metrics.mean_squared_error(pred, np.array([feature])))
        scores.append(score)
    
    # Save scores (anomaly scores)
    df = pd.DataFrame({'anomaly_score': scores, 'is_anomaly': test_labels}) 
    df.to_csv(file_name, index=False)
    tc.green('Saved file {}'.format(file_name))

    visualize_and_save(scores, test_labels, file_name, regularization_strength)


def load_and_show(file_name, regularization_strength, show=True):
    data = pd.read_csv(file_name)
    scores = data.anomaly_score.values
    labels = data.is_anomaly.values
    visualize_and_save(scores, labels, file_name, regularization_strength, show)
    return data

def load_and_label_data(features_fn, threshold, scores_fn):
    features = pd.read_csv(features_fn)
    anomaly_data = pd.read_csv(scores_fn)
    mask = anomaly_data.anomaly_score > threshold
    # Save labelled anomaly scores
    anomaly_data.loc[mask, 'is_anomaly'] = 1
    fn = scores_fn.replace('.csv', '_labelled_{}.csv'.format(
        str(threshold).replace('.', '_')
    ))
    anomaly_data.to_csv(fn, index=False)
    tc.green('Saved file {}'.format(fn))
    # Save labelled features
    features.loc[mask, 'is_anomaly'] = 1
    fn = features_fn.replace('.csv', '_labelled_{}.csv'.format(
        str(threshold).replace('.', '_')
    ))
    features.to_csv(fn, index=False)
    tc.green('Saved file {}'.format(fn))
    # Return anomaly windows
    anomaly_windows = features.loc[mask].window_label
    return anomaly_windows