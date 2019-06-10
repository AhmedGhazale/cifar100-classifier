from tensorflow.keras.datasets import cifar100
import tensorflow as tf
import numpy as np
import cv2
from classifier import Classifier
import config as cfg
from dataset import make_dataset

def shuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

train_data,test_data=cifar100.load_data()

train_img, train_lab = train_data
test_img, test_lab = test_data

#data agumentation
train_img_flip_horz=np.array([cv2.flip(i,1) for i in train_img])
train_img_flip_vert=np.array([cv2.flip(i,0) for i in train_img])
train_img_trans=np.array([cv2.transpose(i) for i in train_img])

train_img=np.append(train_img_flip_horz,train_img,0)
train_img=np.append(train_img_trans,train_img,0)
train_img=np.append(train_img_flip_vert,train_img,0)

train_lab=np.append(train_lab,train_lab,0)
train_lab=np.append(train_lab,train_lab,0)


test_lab = test_lab.reshape([-1])
train_lab = train_lab.reshape([-1])

shuffle(train_img,train_lab)


clasifier = Classifier((cfg.image_size,cfg.image_size,cfg.image_channels),cfg.classes,cfg.model_path)
clasifier.train((train_img,train_lab),(test_img,test_lab),cfg.epochs,cfg.batch_size,cfg.learning_rate,cfg.save_model_path)



