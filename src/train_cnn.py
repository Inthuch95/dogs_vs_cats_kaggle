'''
Created on Jul 18, 2018

@author: User
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from utils import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import get_predictions
from keras import backend as K
import numpy as np 
import os

# dimensions of our images.
img_width, img_height = 150, 150
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 19984
nb_validation_samples = 4992
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
        
def get_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def get_data():
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    return train_generator, validation_generator
    
def train(model, train_generator, validation_generator):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    log_dir = '../CNN/log/'
    callbacks = [ModelCheckpoint('../CNN/cnn.h5', monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir=log_dir, write_graph=True)]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks)
    
def evaluate(validation_generator):
    # evaluate the model with validation set
    y_true = np.array([0] * len(os.listdir('../data/validation/cats/')) + [1] * len(os.listdir('../data/validation/dogs/')))
    model = load_model('../CNN/cnn.h5')
    print(model.summary())
    scores = model.evaluate_generator(validation_generator)
    print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
    y_pred = get_predictions(model, validation_generator)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot percentage confusion matrix
    plot_confusion_matrix(cm_percent, class_names=['Cat', 'Dog'])
    plt.savefig('../MLP/cm_percent_val.png', format='png')
    plt.show()

if __name__ == '__main__':
    train_generator, validation_generator = get_data()
    model = get_network()
#     train(model, train_generator, validation_generator)
    evaluate(validation_generator)