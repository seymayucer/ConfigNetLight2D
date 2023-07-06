# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50

import numpy as np


class RealEncoder(tf.keras.models.Model):
    def __init__(self, latent_dim, input_shape):
        super(RealEncoder, self).__init__()

        self.resnet = resnet50.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            pooling="avg",
        )

        self.resnet_feature_dim = np.prod(self.resnet.output.shape[1:])

        self.feature_to_latent_mlp = keras.layers.Dense(
            latent_dim, input_shape=(self.resnet_feature_dim,)
        )

    def call(self, input_img):
        input_img_0_255 = (input_img + 1) * 127.5
        preprocessed_img = resnet50.preprocess_input(input_img_0_255)

        resnet_features = self.resnet(preprocessed_img)
        embedding = self.feature_to_latent_mlp(resnet_features)

        return embedding
