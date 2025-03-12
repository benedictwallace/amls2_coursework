import tensorflow as tf
from keras import layers, Model, Sequential, optimizers




def discriminator(in_shape):
    model = Sequential()
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding="same", input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout())
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = optimizers.Adam(lr=0.001, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


def generator(gen_init, res_no):
    
    gen = layers.Conv2D(64, (9,9), padding="same")(gen_init)
    gen = layers.PReLU(shared_axes=[1,2])(gen)

    temp = gen

    for i in range(0, res_no):
        gen = residual_block(gen)

    gen = layers.Conv2D(64, (3,3), padding="same")(gen)
    gen = layers.BatchNormalization(momentum=0.5)(gen)
    gen = layers.add([gen, temp])

    gen = upscale(gen)
    gen = upscale(gen)

    op = layers.Conv2D(3, (9,9), padding="same")(gen)

    return Model(inputs=gen_init, outputs=op)
    


def residual_block(initial):
    res = layers.Conv2D(64, (3,3), padding="same")(initial)
    res = layers.BatchNormalization(momentum=0.5)(res)
    res = layers.PReLU(shared_axes = [1,2])(res)
    res = layers.Conv2D(64, (3,3), padding="same")(res)
    res = layers.BatchNormalization(momentum=0.5)(res)

    return layers.add([initial, res])

def upscale(initial):
    up_model = layers.Conv2D(256, (3,3), padding="same")(initial)
    up_model = layers.UpSampling2D(size=2)(initial)
    up_model = layers.PReLU(shared_axes=[1,2])(up_model)

    return up_model