import calendar
import os
import sys
import time

import math
import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adagrad
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value

name_logs = "adagrad"
configure("./logs/" + name_logs, flush_secs=5)


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(100,)))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def train(batch_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    discriminator = discriminator_model()
    generator = generator_model()
    d_on_g = generator_containing_discriminator(generator, discriminator)
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
    generator.compile(loss='mse', optimizer=adagrad)
    d_on_g.compile(loss='mse', optimizer=adagrad)
    discriminator.trainable = True
    discriminator.compile(loss='mse', optimizer=adagrad)
    for epoch in range(25):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / batch_size))
        for index in range(int(X_train.shape[0] / batch_size)):
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                # image = image * 127.5 + 127.5
                # plt.imshow(image)
                # plt.show()
                # path = "C:\\Users\\PycharmProjects\\DCGAN\\train"
                # os.chdir(path)
                # Image.fromarray(image.astype(np.uint8)).save(
                #     str(epoch) + "_" + str(index) + ".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            ts = calendar.timegm(time.gmtime())
            log_value('d_loss', d_loss, ts)
            print("batch %d d_loss : %f" % (index, d_loss))

            noise = np.random.uniform(-1, 1, (batch_size, 100))
            discriminator.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            print("batch %d g_loss : %f" % (index, g_loss))
            log_value('g_loss', g_loss, ts)
            discriminator.trainable = True
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(batch_size, nice=False):
    g = generator_model()
    adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
    g.compile(loss='mse', optimizer=adagrad)
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='mse', optimizer=adagrad)
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (batch_size * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    plt.imshow(image)
    plt.show()
    path = "C:\\Users\\PycharmProjects\\DCGAN\\train\\gene"
    os.chdir(path)
    Image.fromarray(image.astype(np.uint8)).save(name_logs + "generated_image.png")


if __name__ == "__main__":
    train(256)
    generate(256, False)
    sys.exit()
