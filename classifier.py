import tensorflow as tf
import cv2
import numpy as np
import config as cfg


class Classifier:
    def __init__(self, image_shape, num_of_classes, model_path=None):

        self.images = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name="images")
        self.labels = tf.placeholder(tf.float32, [None, num_of_classes], name='labels')
        self.training = tf.placeholder(tf.bool, name='training')
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        self.logits = self.build_model()
        self.sess = tf.Session()

        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

        if model_path == None:
            self.init_weights()
        else:
            self.load_model(model_path)

    def conv_layer(self, input, kernel_size, filters, batch_norm=True, padding="same", activation=tf.nn.relu):
        layer = tf.layers.conv2d(input, filters, [kernel_size, kernel_size], padding=padding)
        if batch_norm:
            layer = tf.layers.batch_normalization(layer, training=True)
        if activation != None:
            layer = activation(layer)
        return layer

    def pool_layer(self, input, pool_size=2, strides=2, type='max'):

        if type == 'max':
            return tf.layers.max_pooling2d(input, [pool_size, pool_size], strides=strides)
        elif type == 'average':
            return tf.layers.average_pooling2d(input, [pool_size, pool_size], strides=strides)

    def dense_layer(self, input, units, drop_out_rate=0.0, activation=None, training=False):
        layer = tf.layers.dense(input, units, activation=activation)
        if (drop_out_rate > 0.0):
            layer = tf.layers.dropout(layer, drop_out_rate, training=training)
        return layer

    def build_model(self):
        X = tf.layers.batch_normalization(self.images, training=True)
        conv1 = self.conv_layer(X, 4, 32)
        conv2 = self.conv_layer(conv1, 3, 64)
        pool1 = self.pool_layer(conv2, type='average')
        conv3 = self.conv_layer(pool1, 3, 128)
        conv4 = self.conv_layer(conv3, 3, 128)
        pool2 = self.pool_layer(conv4, type='average')
        conv5 = self.conv_layer(pool2, 2, 256)
        flat = tf.layers.flatten(conv5)
        dens1 = self.dense_layer(flat, 1024, .5, activation=tf.nn.relu, training=self.training)
        dens2 = self.dense_layer(dens1, 700, .5, activation=tf.nn.relu, training=self.training)

        out = self.dense_layer(dens2, cfg.classes)
        out = tf.identity(out, name="logits")

        return out

    def init_weights(self):
        self.sess.run(tf.global_variables_initializer())

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    def pridect(self, images):

        logits = self.sess.run(self.logits, feed_dict={self.images: images, self.training: False})
        labels = np.argmax(logits, 1)

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
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        test_lab = np.eye(100)[test_labels]
        max_test_accuracy = 0
        for epoch in range(epochs):

            iter_per_epoch = train_images.shape[0] // batch_size

            epoch_loss = 0
            for iter in range(iter_per_epoch):
                x_batch, y_batch = self.get_batch(train_images, train_labels, batch_size, iter)
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict={self.images: x_batch,
                                                   self.labels: y_batch,
                                                   self.training: True,
                                                   self.learning_rate: learning_rate})

                epoch_loss += loss

            test_accuracy = self.get_accuracy(test_images, test_labels)
            print("epoch ", epoch, "with loss", epoch_loss, " test accuracy ", test_accuracy)
            self.shuffle(train_images, train_labels)

            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                self.saver.save(self.sess, save_path)
                print("saved model")

        print("finished training with max test accuracy of ", max_test_accuracy)


