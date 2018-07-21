'''
Created on Jul 16, 2018

@author: User
'''
from sklearn.model_selection import train_test_split
import shutil
import os

images= [img for img in os.listdir('../data/train/') if '.jpg' in img]

def move_files():
    label = []
    for img in images:
        if 'dog' in img:
            label.append(1)
        else:
            label.append(0)
    X_train, X_test, _, _ = train_test_split(images, label, test_size=0.2, random_state=42)
    
    move_to_dir(X_train, 'train')
    print('Moved all training images')  
            
    move_to_dir(X_test, 'validation')
    print('Moved all validation images')
    
def move_to_dir(data, data_type):
    if data_type == 'train':
        src = '../data/train/'
        des_cats = '../data/train/cats/'
        des_dogs = '../data/train/dogs/'
    elif data_type == 'validation':
        src = '../data/validation/'
        des_cats = '../data/validation/cats/'
        des_dogs = '../data/validation/dogs/'
    else:
        print('Unknown data type')
        
    for img in data:
        if 'dog' in img:
            src = src + img
            shutil.move(src, des_dogs)
        else:
            src = src + img
            shutil.move(src, des_cats)  
        
if __name__ == '__main__':
    move_files()