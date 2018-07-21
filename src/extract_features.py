'''
Created on Jul 16, 2018

@author: User
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import numpy as np

model = VGG16(include_top=False, weights='imagenet')
batch_size = 16
img_width, img_height = 150, 150
train_size = 19984
validation_size = 4992
test_size = 12500
train_data_dir = '../data/train/'
validation_data_dir = '../data/validation/'
test_data_dir = '../data/test/'

def extract_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, train_size // batch_size)
    np.save('../bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, validation_size // batch_size)
    np.save('../bottleneck_features_validation.npy', bottleneck_features_validation)
    
def extract_feature_test():
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(generator, test_size // 1)
    print(bottleneck_features_test.shape)
    np.save('../bottleneck_features_test.npy', bottleneck_features_test)

if __name__ == '__main__':
#     extract_features()
    extract_feature_test()