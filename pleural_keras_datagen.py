import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import pandas as pd 
import cv2 
from cv2 import imread, cvtColor
import numpy as np
from keras.applications.densenet import preprocess_input
from keras.models import load_model
from keras import losses
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle


with open("/ctm-hdd-pool01/wjsilva19/MedIA/Train_images_AP_resized/Annotations.pickle",'rb') as fp:
        annotations_train = pickle.load(fp)
        
with open("/ctm-hdd-pool01/wjsilva19/MedIA/Val_images_AP_resized/Annotations.pickle",'rb') as fp:
        annotations_val = pickle.load(fp)


PATH_TRAIN = "/ctm-hdd-pool01/wjsilva19/MedIA/Train_images_AP_resized/"


train_path = [] 
train_clf = [] 
for path, clf in zip(annotations_train[:,0], annotations_train[:,1]): 
    train_path.append(path)
    train_clf.append(int(clf))
    

print(len(train_clf))

print(train_clf.count(1))


df = pd.DataFrame({'Path':train_path, 'Pleural Effusion':train_clf})
print(df)


df['Path'] = df['Path'].astype(str) + '.jpg'

print('Number of elements: ', len(annotations_train))

PATH_VAL = "/ctm-hdd-pool01/wjsilva19/MedIA/Val_images_AP_resized/"

val_imgs = []
val_labels = [] 

for img_path, clf in zip(annotations_val[:,0], annotations_val[:,1]): 
    path = PATH_VAL + img_path + '.jpg'
    img = imread(path)    
    img = cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    val_imgs.append(img)
    val_labels.append(int(clf))


val_imgs = np.asarray(val_imgs)

val_labels = np.asarray(val_labels)
print(val_imgs.shape)
print(val_labels.shape)


def image_preprocessing(img): 
    img = preprocess_input(img)       
    return img

train_datagen = ImageDataGenerator(rotation_range=5, \
                                   width_shift_range=0.05, \
                                   height_shift_range=0.1, \
                                   shear_range=0, \
                                   zoom_range=0, \
                                   horizontal_flip=False, \
                                   preprocessing_function=image_preprocessing)


train_generator=train_datagen.flow_from_dataframe(dataframe=df, \
                                                  directory=PATH_TRAIN, \
                                                  x_col="Path", \
                                                  y_col="Pleural Effusion", \
                                                  batch_size=32, \
                                                  seed=42, \
                                                  shuffle=False, \
                                                  classes=[0,1], \
                                                  class_mode="binary", \
                                                  target_size=(224,224) ,\
                                                  save_to_dir="/ctm-hdd-pool01/wjsilva19/MedIA/Augmented_Images/")

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model = load_model('pleural_dense_AP.hdf5', custom_objects={"auroc": auroc})

# It may make sense to change the optimizer from adadelta to adam(lr=0.0001)
# for smoother learning curve 
model.compile(optimizer='adadelta', loss=losses.binary_crossentropy, \
              metrics=['accuracy', f1])

model.summary()

checkpoint = ModelCheckpoint('/ctm-hdd-pool01/wjsilva19/MedIA/Train_best_f1_miccai_weighted.hdf5', \
                             monitor='val_f1', \
                             save_best_only=True, verbose=True, mode='max')

model.fit_generator(train_generator,
                    steps_per_epoch=len(df.index)/32,
                    epochs=10, \
                    verbose=1, \
                    validation_data=(val_imgs, val_labels),\
                    callbacks=[checkpoint],\
                    class_weight=[train_clf.count(0),train_clf.count(1)])

model.save('/ctm-hdd-pool01/wjsilva19/MedIA/Train_final_f1_miccai_weighted_.hdf5')





    
    
    
    




