import tensorflow as tf
import numpy as np


def parse(x,y):

    x = tf.cast(x,tf.float32)
    x_normalized=x/225
    y_one_hot=tf.one_hot(y,100)

    return x_normalized,y_one_hot

def make_dataset(images,labels,batch_size,epochs,shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images,labels))

    dataset = dataset.map(parse).repeat(epochs).batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset









