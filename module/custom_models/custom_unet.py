import tensorflow as tf

def downsample_layer(x, n_filters, dropout=True):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', 
                               activation='relu')(x)
    skip = tf.keras.layers.Conv2D(n_filters, 3, padding='same', 
                                  activation='relu')(x)

    x = tf.keras.layers.MaxPool2D()(skip)
    if dropout == True:
        x = tf.keras.layers.Dropout(0.2)(x)
    return x, skip

def upsample_layer(x, conv_features, n_filters):
    
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)

    x = tf.keras.layers.concatenate([x, conv_features])

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', 
                               activation='relu')(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding='same', 
                               activation='relu')(x)

    return x

def build_unet(in_shape, output_channels, name):
    
    input = tf.keras.layers.Input(in_shape)

    x, skip1 = downsample_layer(input, 16)
    x, skip2 = downsample_layer(x, 32)
    x, skip3 = downsample_layer(x, 64)
    x, skip4 = downsample_layer(x, 128)
    x, skip5 = downsample_layer(x, 256)

    x = tf.keras.layers.Conv2D(512, 3, padding='same', 
                               activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', 
                               activation='relu')(x)

    x = upsample_layer(x, skip5, 256)
    x = upsample_layer(x, skip4, 128)
    x = upsample_layer(x, skip3, 64)
    x = upsample_layer(x, skip2, 32)
    x = upsample_layer(x, skip1, 16)

    out = tf.keras.layers.Conv2D(output_channels, 3, padding='same', activation='softmax')(x)

    return tf.keras.Model(inputs=input, outputs = out, name=name)
