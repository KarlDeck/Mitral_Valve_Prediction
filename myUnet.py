import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, Activation,\
    BatchNormalization, Dropout, MaxPooling2D,\
    Conv2D, concatenate, Conv2DTranspose

from tensorflow.keras.layers import Multiply, Add, UpSampling2D

## I built a few simple building Blocks below to test different models. In the end what worked best was a regular u net with a depth of 5

def attention_block(skip_features, gating_features, num_filters):
    """Attention block to refine skip connections."""
    # 1x1 Convolutions
    theta = Conv2D(num_filters, kernel_size=3, strides=1, padding="same")(skip_features)  # Skip branch
    phi = Conv2D(num_filters, kernel_size=3, strides=1, padding="same")(gating_features)  # Gating branch

    # Add and apply ReLU activation
    attention = Add()([theta, phi])
    attention = Activation("relu")(attention)

    # Attention coefficients
    psi = Conv2D(1, kernel_size=1, strides=1, padding="same")(attention)
    psi = Activation("sigmoid")(psi)

    # Multiply skip features with attention coefficients
    refined_skip = Multiply()([skip_features, psi])
    return refined_skip

def conv_block(inputs, activation_funct, num_filters, dropout):
    """Basic convolutional block: Conv -> activation_funct -> Conv -> activation_funct"""
    x = Conv2D(num_filters, 3, kernel_initializer='he_normal', padding="same", activation=activation_funct)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Conv2D(num_filters, 3, kernel_initializer='he_normal', padding="same", activation=activation_funct)(x)
    x = BatchNormalization()(x)
    return x

def encoder_block(inputs, activation_funct, num_filters, dropout):
    """Encoder block: Conv block -> MaxPooling"""
    x = conv_block(inputs, activation_funct, num_filters, dropout)
    p = MaxPooling2D(2)(x)
    return x, p

def upsample_block(inputs, skip_features, activation_funct, num_filters, dropout):
    """Upsampling block: Upsample -> Concatenate -> Conv block"""
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    # Apply attention to skip features
    #refined_skip = attention_block(skip_features, x, num_filters)
    # Concatenate with refined skip connection
    x = concatenate([x, skip_features])
    x = conv_block(x, activation_funct, num_filters, dropout)
    return x

def test_build_myunet(input_shape=(128, 128, 1), num_classes=1, base_filters=64, activation_funct = "relu", dropout=0.3):
    #This was used as a testing ground for alternative structures e.g. using skip or attention --> did not result in better predictions
    """Experimental U-net with ROI(Region of interest) prediction"""
    #The ROI prediction did not help and was way slower

    # Encoder
    inputs = Input(input_shape)
    enc1, pool1 = encoder_block(inputs, activation_funct, base_filters, dropout)    # 128x128 -> 64x64
    enc2, pool2 = encoder_block(pool1, activation_funct, base_filters*2, dropout)    # 64x64 -> 32x32
    enc3, pool3 = encoder_block(pool2, activation_funct, base_filters*4, dropout)    # 32x32 -> 16x16
    enc4, pool4 = encoder_block(pool3, activation_funct, base_filters*8, dropout)    # 16x16 -> 8x8
    enc5, pool5 = encoder_block(pool4, activation_funct, base_filters*16, dropout)    # 8x8 -> 4x4


    # Bottleneck
    bottleneck = conv_block(pool5, activation_funct, base_filters*32, dropout)       # 8x8

    # Decoder
    dec5 = upsample_block(bottleneck, enc5, activation_funct, base_filters*16, dropout)  # 4x4 -> 8x8
    dec4 = upsample_block(dec5, enc4, activation_funct, base_filters*8, dropout)       # 8x8 -> 16x16
    dec3 = upsample_block(dec4, enc3, activation_funct, base_filters*4, dropout)       # 16x16 -> 32x32
    dec2 = upsample_block(dec3, enc2, activation_funct, base_filters*2, dropout)       # 32x32 -> 64x64
    dec1 = upsample_block(dec2, enc1, activation_funct, base_filters, dropout)        # 64x64 -> 128x128

    # Output layer
    outputs_segmentation = Conv2D(num_classes, 1, activation="sigmoid", name="outputs_segmentation")(dec1)
    outputs_roi = Conv2D(1, kernel_size=1, activation="sigmoid", name="outputs_roi")(dec1)  # ROI output

    model = Model(inputs, [outputs_segmentation, outputs_roi])

    return model


def build_myunet(input_shape=(128, 128, 1), num_classes=1, base_filters=64, activation_funct = "relu", dropout=0.1):
    """Ordinary U-Net model without pretrained encoder."""
    # Encoder
    inputs = Input(input_shape)
    enc1, pool1 = encoder_block(inputs, activation_funct, base_filters, dropout)    # 128x128 -> 64x64
    enc2, pool2 = encoder_block(pool1, activation_funct, base_filters*2, dropout)    # 64x64 -> 32x32
    enc3, pool3 = encoder_block(pool2, activation_funct, base_filters*4, dropout)     # 32x32 -> 16x16
    enc4, pool4 = encoder_block(pool3, activation_funct, base_filters*8, dropout)      # 16x16 -> 8x8

    # Bottleneck
    bottleneck = conv_block(pool4, activation_funct, base_filters*32, dropout)          # 8x8

    # Decoder
    dec4 = upsample_block(bottleneck, enc4, activation_funct, base_filters*8, dropout) # 8x8 -> 16x16
    dec3 = upsample_block(dec4, enc3, activation_funct, base_filters*4, dropout)      # 16x16 -> 32x32
    dec2 = upsample_block(dec3, enc2, activation_funct, base_filters*2, dropout)     # 32x32 -> 64x64
    dec1 = upsample_block(dec2, enc1, activation_funct, base_filters, dropout)      # 64x64 -> 128x128

    # Output layer
    outputs = Conv2D(num_classes, 1, activation="sigmoid")(dec1)

    # Build the U-Net model
    model = Model(inputs, outputs)

    return model