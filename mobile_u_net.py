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

    mobile_net = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, weights='imagenet', include_top=False
    )
    mobile_net.trainable = False

    skip_4 = mobile_net.get_layer("block_13_expand_relu").output
    skip_3 = mobile_net.get_layer("block_6_expand_relu").output
    skip_2 = mobile_net.get_layer("block_3_expand_relu").output
    skip_1 = mobile_net.get_layer("block_1_expand_relu").output
    bottleneck = mobile_net.output

    encoder = tf.keras.Model(
        inputs = mobile_net.input,
        outputs = [skip_1, skip_2, skip_3, skip_4, bottleneck],
        name = "mobile_net"
    )

    skip_1, skip_2, skip_3, skip_4, x = encoder(inputs)

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

    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mobile_u_net')
    return model

def flip(image, mask):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask


if __name__ == "__main__":
    model = build_mobile_u_net(input_shape=(128,384,3), num_classes=20)
    model.summary()
