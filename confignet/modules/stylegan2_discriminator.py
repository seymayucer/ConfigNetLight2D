"""
@author seyma
@date 10.11.2022
@ref :https://raw.githubusercontent.com/cryu854/StyleGAN2/main/modules/discriminator.py 
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
from .layers.conv2d import conv2d
from .layers.fully_connected import fully_connected


class dis_block(tf.keras.layers.Layer):
    def __init__(self, fmap, fmap_down, impl="ref", **kwargs):
        super(dis_block, self).__init__(**kwargs)
        self.conv = conv2d(fmap, kernel_size=3, impl=impl)
        self.conv_down = conv2d(fmap_down, kernel_size=3, down=True, impl=impl)
        self.conv_skip = conv2d(
            fmap_down,
            kernel_size=1,
            down=True,
            apply_bias=False,
            apply_lrelu=False,
            impl=impl,
        )

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.conv_down(x)
        residual = self.conv_skip(inputs)
        return (x + residual) * tf.math.rsqrt(2.0)


# Minibatch standard deviation layer at the end of the discriminator
class minibatch_stddev(tf.keras.layers.Layer):
    def __init__(self, group_size=4, num_new_features=1, **kwargs):
        super(minibatch_stddev, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, inputs, training=None):
        x = inputs
        s = tf.shape(x)  # [NHWC]  Input shape.
        group_size = tf.minimum(self.group_size, s[0])
        y = tf.reshape(
            x,
            [
                group_size,
                -1,
                s[1],
                s[2],
                s[3] // self.num_new_features,
                self.num_new_features,
            ],
        )  # [GMHWnc] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)  # [GMHWcn] Cast to FP32.
        y -= tf.reduce_mean(
            y, axis=0, keepdims=True
        )  # [GMHWcn] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MHWcn]  Calc variance over group.
        y = tf.math.sqrt(y + 1e-8)  # [MHWcn]  Calc stddev over group.
        y = tf.reduce_mean(
            y, axis=[1, 2, 3], keepdims=True
        )  # [M111n]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[3])  # [M11n] Split channels into c channel groups
        y = tf.cast(y, x.dtype)  # [M11n]  Cast back to original data type.
        y = tf.tile(
            y, [group_size, s[1], s[2], 1]
        )  # [NHWn]  Replicate over group and pixels.
        return tf.concat([x, y], axis=-1)  # [NHWC]  Append as new fmap.


class StyleGAN2Discriminator(Model):
    def __init__(
        self,
        resolution,
        config,
        impl="ref",
        mbstd_group_size=4,
        mbstd_num_features=1,
        name="discriminator",
    ):

        super(StyleGAN2Discriminator, self).__init__()
        res_index = int(np.log2(resolution)) - 2
        filter_multiplier = 2 if config == "f" else 1
        filters = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * filter_multiplier,
            128: 128 * filter_multiplier,
            256: 64 * filter_multiplier,
            512: 32 * filter_multiplier,
            1024: 16 * filter_multiplier,
        }
        self.initial_conv_layer = conv2d(
            filters[resolution],
            kernel_size=1,
            name=f"from_rgb_{resolution}x{resolution}",
        )
        self.blocks = []

        for res, fmaps in list(filters.items())[res_index:0:-1]:
            fmaps_down = filters[res / 2]
            self.blocks.append(
                dis_block(fmaps, fmaps_down, impl=impl, name=f"{res}x{res}")
            )

        self.minibatch_stddev = minibatch_stddev(
            mbstd_group_size, mbstd_num_features, name="mb_std"
        )

        self.second_conv_layer = conv2d(
            filters=filters[8], kernel_size=3, impl=impl, name="conv_4x4"
        )

        self.fc1 = fully_connected(units=filters[4], name="fc_4x4")

        self.fc2 = fully_connected(units=1, apply_lrelu=False, name="fc_out")
        self.build((None, resolution, resolution, 3))

    def call(self, input_img):
        x = self.initial_conv_layer(input_img)
        for bb in self.blocks:
            x = bb(x)
        x = self.minibatch_stddev(x)
        x = self.second_conv_layer(x)
        latents = self.fc1(x)
        out = self.fc2(latents)
        return out, latents
