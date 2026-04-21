import tensorflow as tf
from tensorflow.keras import layers, Model


def build_unet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # encoder
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    skip_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(skip_1)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    skip_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(skip_2)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    skip_3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(skip_3)

    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    skip_4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(skip_4)

    x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip_4])
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip_3])
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip_2])
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, skip_1])
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(num_classes, (1,1), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='unet')
    return model