'''
Created on Jul 16, 2018

@author: User
'''
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os

def load_data():
    X_train = np.load('../bottleneck_features_train.npy') 
    print(X_train.shape)
    y_train = np.array([0] * len(os.listdir('../data/train/cats/')) + [1] * len(os.listdir('../data/train/dogs/')))
    
    X_test = np.load('../bottleneck_features_validation.npy') 
    print(X_test.shape)
    y_test = np.array([0] * len(os.listdir('../data/validation/cats/')) + [1] * len(os.listdir('../data/validation/dogs/')))
    
    return X_train, X_test, y_train, y_test

def load_test_data():
    X = np.load('../bottleneck_features_test.npy') 
    print(X.shape)
    
    return X

def plot_confusion_matrix(cm, title='Confusion matrix', float_display='.4f', cmap=plt.cm.Greens, class_names=None):
    # create confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    plt.yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], float_display),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
def get_predictions(model, X_test):
    y_pred = []
    predictions = model.predict_generator(X_test)
    for pred in predictions:
        if pred >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

if __name__ == '__main__':
    pass