'''
Created on Jul 16, 2018

@author: User
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils import load_test_data
import pickle
import numpy as np
import pandas as pd

def submit_mlp(X):
    model = load_model('../MLP/mlp.h5')
    predictions = model.predict(X)
    label = []
    for pred in predictions:
        label.append(pred[0])
    index = np.array([i for i in range(1, len(label)+1)])
    df = pd.DataFrame(index, columns=['id'])
    df['label'] = label
    print(df.head())
    df.to_csv('../submission_mlp.csv', index=False)
    
def submit_svm(X):
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    model = pickle.load(open('../SVM/svm.pkl', 'rb'))
    label = model.predict(X)
    index = np.array([i for i in range(1, len(label)+1)])
    df = pd.DataFrame(index, columns=['id'])
    df['label'] = label
    print(df.head())
    df.to_csv('../submission_svm.csv', index=False)
    
def submit_cnn():
    model = load_model('../CNN/cnn.h5')
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        '../data/test/',
        target_size=(150, 150),
        batch_size=1,
        class_mode='binary')
    predictions = model.predict_generator(test_generator, 12500 // 1)
    label = []
    for pred in predictions:
        label.append(pred[0])
    index = np.array([i for i in range(1, len(label)+1)])
    df = pd.DataFrame(index, columns=['id'])
    df['label'] = label
    print(df.head())
    df.to_csv('../submission_cnn.csv', index=False)

if __name__ == '__main__':
    X = load_test_data()
#     submit_mlp(X)
#     submit_svm(X)
    submit_cnn()