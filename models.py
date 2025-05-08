from tensorflow.keras import layers, Sequential

def build_generator(latent_dim=100, generator_filters=64, dropout_rate=0.3):
    # Building a Generator model
    model = Sequential()
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(8*8*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(generator_filters*2, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2DTranspose(generator_filters, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator(discriminator_filters=64, dropout_rate=0.3):
    #Building a Discriminator model
    model = Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Conv2D(discriminator_filters, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Conv2D(discriminator_filters*2, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model