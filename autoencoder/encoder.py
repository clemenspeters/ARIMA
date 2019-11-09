'''
 # Source: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_14_03_anomaly.ipynb
'''

from sklearn import metrics
import numpy as np
import pandas as pd
# from IPython.display import display, HTML 
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


def run(file_name_training, file_name_test):
    # Load data from files
    features_training = pd.read_csv('./data/{}.csv'.format(file_name_training))
    features_test = pd.read_csv('./data/{}.csv'.format(file_name_test))
    # Format data. TODO: refactor
    x_normal = np.array(features_training)
    x_normal_train = np.array(features_training)
    x_normal_test = np.array(features_test)


    # Train autoencoder network
    model = Sequential()
    model.add(Dense(25, input_dim=x_normal.shape[1], activation='relu'))
    model.add(Dense(2, activation='relu'))
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
    df = pd.DataFrame(predictions) 
    df.to_csv('results/autoencoder_anomaly_scores_100_1600.csv', header=False, index=False) 

    # Visualize predictions
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.plot(predictions) # plotting by columns
    plt.savefig('results/autoencoder_anomaly_scores_100_1600.png')
    plt.show()