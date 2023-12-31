import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def unet_model(output_channels:int):
    SIZE = 224
    base_model = tf.keras.applications.MobileNetV2(input_shape=[SIZE, SIZE, 3], include_top=False)
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input



# Use the activations of these layers
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False


    up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


    inputs = tf.keras.layers.Input(shape=[SIZE, SIZE, 3])
    inputs = preprocess_input(inputs)


    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)
    x = tf.keras.layers.Conv2D(output_channels, (3, 3), activation='softmax', padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

if __name__ == '__main__':
    unet_model(24)