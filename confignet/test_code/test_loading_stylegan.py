import tensorflow as tf
import numpy as np
from modules.stylegan_generator import generator
from modules.stylegan_discriminator import discriminator
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def build_model(resolution=256, num_labels=0, config="e", impl="cuda"):
    """ Build initial model """
    D = discriminator(resolution, num_labels, config, impl)
    G = generator(resolution, num_labels, config, impl)
    Gs = generator(resolution, num_labels, config, impl, randomize_noise=False,)
    # Setup Gs's weights same as G
    Gs.set_weights(G.get_weights())
    print(
        "G_trainable_parameters:",
        np.sum([np.prod(v.get_shape().as_list()) for v in G.trainable_variables]),
    )
    print(
        "D_trainable_parameters:",
        np.sum([np.prod(v.get_shape().as_list()) for v in D.trainable_variables]),
    )
    return D, G, Gs


def load_generator_discriminator_weights():
    pl_mean = tf.Variable(
        name="pl_mean", initial_value=0.0, trainable=False, dtype=tf.float32
    )
    elapsed_time = tf.Variable(
        name="elapsed_time", initial_value=0, trainable=False, dtype=tf.int32
    )

    step = tf.Variable(name="step", initial_value=0, trainable=False, dtype=tf.int32)

    max_steps = 100000
    D, G, Gs = build_model()
    ckpt = tf.train.Checkpoint(
        step=step,
        elapsed_time=elapsed_time,
        generator=G,
        discriminator=D,
        generator_clone=Gs,
        pl_mean=pl_mean,
    )
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory="checkpoint", max_to_keep=2
    )

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(
            f"Restored from {ckpt_manager.latest_checkpoint} at step {ckpt.step.numpy()}."
        )
        if step.numpy() >= max_steps:
            print("Training has already completed.")
            return
    else:
        print("Initializing from scratch...")

    breakpoint()


load_generator_discriminator_weights()
