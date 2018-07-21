'''
Created on Jul 16, 2018

@author: User
'''
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from utils import load_data, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

epochs = 50
batch_size = 16

def get_network(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
 
def train(model, X_train, y_train, X_test, y_test):    
    log_dir = '../MLP/log/'
    callbacks = [ModelCheckpoint('../MLP/mlp.h5', monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir=log_dir, write_graph=True)]
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              validation_data=(X_test, y_test))
    return model

def evaluate(X_test, y_test):
    # evaluate the model with validation set
    model = load_model('../MLP/mlp.h5')
    scores = model.evaluate(X_test, y_test)
    print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    plot_confusion_matrix(cm_percent, class_names=['Cat', 'Dog'])
    plt.savefig('../MLP/cm_percent_val.png', format='png')
    plt.show()
    
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = get_network(X_train.shape[1:])
    model = train(model, X_train, y_train, X_test, y_test)
    evaluate(X_test, y_test)
    