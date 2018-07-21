'''
Created on Jul 16, 2018

@author: User
'''
from sklearn.svm import LinearSVC 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from utils import load_data, plot_confusion_matrix
import numpy as np
import pickle
import os

def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    # save SVM model
    name = 'svm.pkl'
    file = os.path.join('../SVM/', name)
    with open(file, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_cv(model, X_train, y_train):
    # evaluate perfromance with 10-fold cv
    scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
    filename = 'svm_cv.pkl'
    file = os.path.join('../SVM/', filename)
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def display_score(scores):
    print('Scores: ', scores)
    print('Accuracy: %0.4f' % (scores.mean()))

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    X_train =X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
    model = LinearSVC()
    evaluate_cv(model, X_train, y_train)
    model = train(model, X_train, y_train)
    
    model = pickle.load(open('../SVM/svm.pkl', 'rb'))
    scores = pickle.load(open('../SVM/'+'svm_cv.pkl', 'rb')) 
    display_score(scores)
    print('Test accuracy: %0.4f' % (model.score(X_test, y_test)))
    print(model.decision_function(X_test))
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, title='SVM', class_names=['Cat', 'Dog'])
    plt.savefig('../SVM/confusion_matrix_svm.png', format='png')
    plt.show()
    