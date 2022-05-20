# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os
import numpy as np


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""

    nb_features = x_train.shape[2]
    sequence_length = x_train.shape[1]
    nb_out = y_train.shape[1]
    
    model = Sequential()
    
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='RMSProp', metrics=[tf.keras.metrics.AUC()])

    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)

    return model


def _load_training_data(base_dir):
    """Load training data"""
    x_train = np.load(os.path.join(base_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load testing data"""
    x_test = np.load(os.path.join(base_dir, 'x_val.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_val.npy'))
    return x_test, y_test

    return parser.parse_known_args()


if __name__ == "__main__":
    # define path
    prefix = '/opt/ml/'
    
    input_path = prefix + 'input/data'
    output_path = os.path.join(prefix, 'output')
    model_path = os.path.join(prefix, 'model')

    channel_name = 'train'
    train_path = os.path.join(input_path, channel_name)
    
    # load data
    train_data, train_labels = _load_training_data(train_path)
    eval_data, eval_labels = _load_testing_data(train_path)
    
    #model training
    tf_classifier = model(train_data, train_labels, eval_data, eval_labels)

    #save model to an S3 directory with vesion number '000000001'
    tf_classifier.save(os.path.join(model_path,'000000001'), 'tf-model.h5')
 
    
    
