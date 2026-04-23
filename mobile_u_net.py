import numpy as np
import tensorflow as tf

from data.devkit_semantics.devkit.helpers.labels import labels

mapping = np.full(256, 255, dtype=np.int32)
for label in labels:
    if label.id >= 0:
        mapping[label.id] = label.trainId
mapping[mapping == 255] = 19

def load_preprocess_mobilenet(image_path, mask_path):
    raw_image = tf.io.read_file(image_path)
    raw_mask = tf.io.read_file(mask_path)

    decoded_image = tf.image.decode_png(raw_image, channels=3)
    decoded_mask = tf.image.decode_png(raw_mask, channels=1)

    resized_image = tf.image.resize(decoded_image, [128,384])
    resized_mask = tf.image.resize(decoded_mask, [128,384], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    label_mapping = tf.constant(mapping, dtype=tf.int32)
    resized_mask = tf.gather(label_mapping, tf.cast(resized_mask, tf.int32))

    resized_image = tf.keras.applications.mobilenet_v2.preprocess_input(resized_image)

    return resized_image, resized_mask

def build_mobile_u_net(input_shape, num_classes):
    inputs = tf.keras.Input(input_shape)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
    mobile_net.trainable = False

    x = mobile_net(inputs)

    x = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1))(x)
    print(outputs.shape)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mobile_u_net')
    return model


if __name__ == "__main__":
    build_mobile_u_net(input_shape=(128,384,3), num_classes=20)
