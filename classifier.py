import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import config as cfg


class Classifier:
    def __init__(self, image_shape, num_of_classes, model_path=None):

        self.model = self.build_model()
        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

        if model_path == None:
            self.init_weights()
        else:
            self.load_model(model_path)

    def conv_layer(self, input, kernel_size, filters, batch_norm=True, padding="same", activation='relu'):
        layer = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)(input)
        if batch_norm:
            layer = layers.BatchNormalization()(layer)
        return layer

    def pool_layer(self, input, pool_size=2, strides=2, type='max'):
def pool_layer(self, input, pool_size=2, strides=2, type='max'):
    if type == 'max':
        return layers.MaxPooling2D(pool_size, strides)(input)
    elif type == 'average':
        return layers.AveragePooling2D(pool_size, strides)(input)

    def dense_layer(self, input, units, drop_out_rate=0.0, activation=None):
        layer = layers.Dense(units, activation=activation)(input)
        if (drop_out_rate > 0.0):
            layer = layers.Dropout(drop_out_rate)(layer)
        return layer

    def build_model(self):
        inputs = keras.Input(shape=(cfg.image_size, cfg.image_size, cfg.image_channels))
        X = layers.BatchNormalization()(inputs)
        conv1 = self.conv_layer(X, 4, 32)
        conv2 = self.conv_layer(conv1, 3, 64)
        pool1 = self.pool_layer(conv2, type='average')
        conv3 = self.conv_layer(pool1, 3, 128)
        conv4 = self.conv_layer(conv3, 3, 128)
        pool2 = self.pool_layer(conv4, type='average')
        conv5 = self.conv_layer(pool2, 2, 256)
        flat = layers.Flatten()(conv5)
        dens1 = self.dense_layer(flat, 1024, .5, activation='relu')
        dens2 = self.dense_layer(dens1, 700, .5, activation='relu')
        outputs = self.dense_layer(dens2, cfg.classes)
        return keras.Model(inputs=inputs, outputs=outputs)

        return out

    # No need for this function in TensorFlow 2.x

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def pridect(self, images):
def predict(self, images):
    logits = self.model.predict(images)
    labels = np.argmax(logits, 1)
    return labels
        return labels

    def get_accuracy(self, images, lables):

        prid_labels = self.pridect(images)
        correct = np.equal(prid_labels, lables)
        correct = np.array(correct, np.float32)
        accuracy = np.sum(correct) / correct.shape[0]

        return accuracy

    def get_batch(self, images, labels, batch_size, idx):
        batch_x = images[batch_size * idx:batch_size * (idx + 1)]
        batch_y = labels[batch_size * idx:batch_size * (idx + 1)]
        batch_y = np.eye(100)[batch_y]
        return batch_x, batch_y

    def shuffle(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, train_data, test_data, epochs, batch_size, learning_rate, save_path, ):

        # dataset
        def train(self, train_data, test_data, epochs, batch_size, learning_rate, save_path, ):
            # dataset
            train_images, train_labels = train_data
            test_images, test_labels = test_data
            train_labels = np.eye(cfg.classes)[train_labels]
            test_labels = np.eye(cfg.classes)[test_labels]
            history = self.model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))
            self.model.save_weights(save_path)
            print("finished training with max test accuracy of ", max(history.history['val_accuracy']))
                self.saver.save(self.sess, save_path)
                print("saved model")

        print("finished training with max test accuracy of ", max_test_accuracy)

