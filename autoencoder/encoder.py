'''
 # Source: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_14_03_anomaly.ipynb
'''

from sklearn import metrics
import numpy as np
import pandas as pd
# from IPython.display import display, HTML 
import tensorflow as tf
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
    file_path = 'results/{}.png'.format(file_name)
    plt.savefig(file_path)
    plt.clf()
    print('Saved image {}'.format(file_path))
    # plt.show()


def run(file_name_training, file_name_test, regularization_strength):
    # Load data from files
    features_training = pd.read_csv('./data/{}.csv'.format(file_name_training))
    features_test = pd.read_csv('./data/{}.csv'.format(file_name_test))
    # Format data. TODO: refactor
    x_normal = np.array(features_training)
    x_normal_train = np.array(features_training)
    x_normal_test = np.array(features_test)


    # Train autoencoder network
    encoding_dim = 2
    model = Sequential()
    # Input layer
    model.add(Dense(25, input_dim=x_normal.shape[1], activation='relu'))
    # Hidden layer
    model.add(Dense(
        encoding_dim,
        activation='relu',
        activity_regularizer=l1(regularization_strength)
    ))
    # Output layer
    model.add(Dense(25, activation='relu'))
    model.add(Dense(x_normal.shape[1])) # Multiple output neurons
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_normal_train, x_normal_train, verbose=1, epochs=40)

    pred = model.predict(x_normal_train)
    score1 = np.sqrt(metrics.mean_squared_error(pred,x_normal_train))
    print("Training Normal Score (RMSE): {}".format(score1))

    pred = model.predict(x_normal_test)
    score1 = np.sqrt(metrics.mean_squared_error(pred,x_normal_test))
    print("Test Normal Score (RMSE): {}".format(score1))


    # Predict / create anomaly scores
    count = 0
    predictions = []
    for feature in tqdm(x_normal_test):
        pred = model.predict(np.array([feature]))
        score1 = np.sqrt(metrics.mean_squared_error(pred,np.array([feature])))
        predictions.append(score1)
        count += 1
    
    # Save predictions (anomaly scores)
    # result_file_name = 'autoencoder_anomaly_scores_1000_1600'
    result_file_name = 'autoencoder_anomaly_scores_same_train_test_regularization_dense_{}'.format(str(regularization_strength).replace('.', '_'))
    df = pd.DataFrame(predictions) 
    df.to_csv('results/{}.csv'.format(result_file_name), header=False, index=False) 

    visualize_and_save(predictions, result_file_name, regularization_strength)


def load_and_show(file_name, regularization_strength):
    data = pd.read_csv('results/{}.csv'.format(file_name))
    visualize_and_save(data, file_name, regularization_strength)