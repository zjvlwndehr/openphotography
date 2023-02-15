### DC-GAN ###

import keras
from keras.optimizer_v2.adam import Adam
import numpy as np

from pathlib import Path
from lib.io import Train
from datetime import datetime

SHAPE = (64, 64, 3) # (height, width, channels)
BATCH_SIZE = 32
EPOCHS = 10

LEARNING_RATE = 0.0002
Z_DIM = 100
                
def Generator(z_dim : int = 100):
    # input: random noise (z_dim, )
    # output: image (SHAPE[0], SHAPE[1], SHAPE[2])
    model = keras.Sequential()
    model.add(keras.layers.Dense(16 * 16 * 256 * 3, input_dim=z_dim))
    model.add(keras.layers.Reshape((16, 16, 256 * 3)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='tanh'))
    
    return model

def Discriminator(input_shape : tuple = SHAPE):
    # input: image (SHAPE[0], SHAPE[1], SHAPE[2])
    # output: probability (0 ~ 1.0)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid')) # probability (0 ~ 1.0)

    return model

def GAN(generator : keras.Model, discriminator : keras.Model):
    # input: fake image or real image (SHAPE[0], SHAPE[1], SHAPE[2])
    # output: probability (0 ~ 1.0)
    model = keras.Sequential()
    model.add(generator)

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE, beta_1=0.5))
    discriminator.trainable = False
    
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE, beta_1=0.5))

    return model

class Engine():
    def __init__(self, train_path : Path = None, test_path : Path = None, ext : str = 'jpg', epochs : int = 10, batch_size : int = 32, size : tuple = None, crop : tuple = None, rotate : int = None, flip : int = None):
        self.Train = Train(train_path, ext)
        self.model_name = f"DC-GAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_path = Path('checkpoint') / self.model_name
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint = keras.callbacks.ModelCheckpoint(str(self.checkpoint_path / self.model_name), save_weights_only=True, save_best_only=True, verbose=1, monitor='loss', mode='min')

        print('[INFO]: Training data loaded.')
        
    def gen_image(self, count : int = BATCH_SIZE, size = (SHAPE[0], SHAPE[1]), crop = None, rotate = None, flip = None):
        return self.Train.get_images(count, size, crop, rotate, flip)
    
    def train(self, GAN : keras.Sequential = GAN(Generator(), Discriminator()), epochs : int = EPOCHS, batch_size : int = BATCH_SIZE, size : tuple = None, crop : tuple = None, rotate : int = None, flip : int = None) -> None:
        for epoch in range(epochs):
            print(f'[INFO]: Epoch {epoch + 1}/{epochs}')
            for batch in self.gen_image(batch_size, size, crop, rotate, flip):
                print(f'[INFO]: Single Batch {batch.shape}')


if __name__ == '__main__':
    print('''[INFO]: This is a module. To use it, import it.
\tIf you want to train a model, execute train.py.''')

# if __name__ == '__main__':
#     assert False, '[INFO]: This is a module. To use it, import it.'