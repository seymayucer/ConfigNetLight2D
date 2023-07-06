# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" Scripts containing common code for training """
import os
import numpy as np
import tensorflow as tf
import random


def initialize_random_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Directory {dir} createrd")
    else:
        print(f"Directory {dir} already exists")

    return dir


def imsave(image, path):
    image = (image + 1) * 127.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
    image.save(path)
