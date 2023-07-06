# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from tensorflow import keras

pl_mean = tf.Variable(
    name="pl_mean", initial_value=0.0, trainable=False, dtype=tf.float32
)


def GAN_G_loss(scores):
    return keras.backend.mean(tf.math.softplus(-scores))


def GAN_D_loss(labels, scores):
    return keras.backend.mean(
        labels * tf.math.softplus(-scores) + (1.0 - labels) * tf.math.softplus(scores)
    )


def eye_loss(gt_imgs, gen_imgs, eye_masks):
    # breakpoint()
    eye_masks = tf.cast(eye_masks, tf.float32)
    img_diff = (gt_imgs - gen_imgs) * tf.expand_dims(eye_masks, -1)

    loss_val_per_img = tf.reduce_sum(tf.square(img_diff), axis=(1, 2, 3)) / (
        1 + tf.math.reduce_sum(eye_masks, (1, 2))
    )

    return tf.reduce_mean(loss_val_per_img)


def compute_discriminator_loss(discriminator, real_imgs, fake_imgs):
    # Labels
    valid_y = tf.ones((real_imgs.shape[0], 1), dtype=tf.float32)
    fake_y = tf.zeros((fake_imgs.shape[0], 1), dtype=tf.float32)

    losses = {}
    with tf.GradientTape(persistent=True) as grad_reg_tape:
        grad_reg_tape.watch(real_imgs)
        discriminator_output_real, _ = discriminator(real_imgs)
    discriminator_output_fake, _ = discriminator(fake_imgs)

    # GAN loss on real
    for i, (disc_out) in enumerate(discriminator_output_real.values()):
        gan_loss = GAN_D_loss(valid_y, disc_out)
        losses["GAN_loss_real_" + str(i)] = gan_loss

    # GAN loss on fake
    for i, disc_out in enumerate(discriminator_output_fake.values()):
        gan_loss = GAN_D_loss(fake_y, disc_out)
        losses["GAN_loss_fake_" + str(i)] = gan_loss

    # Gradient penalty
    for i, single_discr_output in enumerate(discriminator_output_real.values()):
        losses["gp_loss_" + str(i)] = gradient_regularization(
            grad_reg_tape, single_discr_output, real_imgs
        )

    losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

    return losses


def compute_stylegan_discriminator_loss(
    discriminator, real_imgs, fake_imgs, gamma, compute_reg=True
):
    fake_scores, _ = discriminator(fake_imgs, training=True)
    losses = {}
    reg = 0.0
    if compute_reg:
        with tf.GradientTape(watch_accessed_variables=False) as r1_tape:
            r1_tape.watch(real_imgs)
            real_scores, _ = discriminator(real_imgs, training=True)
            real_loss = tf.reduce_sum(real_scores)

        real_grads = r1_tape.gradient(real_loss, real_imgs)
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        reg = gradient_penalty * (gamma * 0.5)
    else:
        real_scores, _ = discriminator(real_imgs, training=True)
    if real_scores.shape != fake_scores.shape:
        print(real_scores.shape, fake_scores.shape, real_imgs.shape[0])
        breakpoint()
    loss = tf.math.softplus(fake_scores) + tf.math.softplus(-real_scores)
    loss = tf.reduce_mean(loss)
    losses["loss_sum"] = loss
    return losses, reg


@tf.function
def compute_stylegan_generator_loss(
    generator, fake_scores, pl_decay=0.01, pl_weight=2.0, pl_batch_shrink=2
):

    loss = tf.nn.softplus(-fake_scores)
    loss = tf.reduce_mean(loss)

    # Path length regularization.

    if pl_batch_shrink > 1:
        pl_minibatch = fake_scores.shape[0] // pl_batch_shrink
        pl_latents = tf.random.normal([pl_minibatch, 512])
        # pl_labels = tf.zeros([pl_minibatch, 0])

        fake_images_out, fake_dlatents_out = generator(
            pl_latents, return_latents=True, training=True
        )

        # Compute |J*y|.
        pl_noise = tf.random.normal(tf.shape(input=fake_images_out)) / tf.math.sqrt(
            tf.math.reduce_prod(float(generator.resolution))
        )

        # ys = tf.reduce_sum(input_tensor=fake_images_out * pl_noise)

        # pl_grads = tape.gradient(ys, fake_dlatents_out)
        pl_grads = tf.gradients(
            ys=tf.reduce_sum(input_tensor=fake_images_out * pl_noise),
            xs=[fake_dlatents_out],
        )[0]

        pl_lengths = tf.sqrt(
            tf.reduce_mean(
                input_tensor=tf.reduce_sum(input_tensor=tf.square(pl_grads), axis=2),
                axis=1,
            )
        )

        # Track exponential moving average of |J*y|.
        new_pl_mean = pl_mean + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean)
        pl_mean.assign(new_pl_mean)

        # Calculate (|J*y|-a)^2.
        pl_penalty = tf.square(pl_lengths - new_pl_mean)

        reg = pl_penalty * pl_weight

    return loss, reg


def compute_latent_discriminator_loss(latent_discriminator, real_latents, fake_latents):
    batch_size = real_latents.shape[0]

    # Labels
    valid_y = tf.ones((batch_size, 1))
    fake_y = tf.zeros((batch_size, 1))

    losses = {}
    with tf.GradientTape(persistent=True) as grad_reg_tape:
        grad_reg_tape.watch(real_latents)
        discriminator_output_real = latent_discriminator(real_latents)
    discriminator_output_fake = latent_discriminator(fake_latents)

    # GAN loss on real
    gan_loss = GAN_D_loss(valid_y, discriminator_output_real)
    losses["GAN_loss_real"] = gan_loss
    # GAN loss on fake
    gan_loss = GAN_D_loss(fake_y, discriminator_output_fake)
    losses["GAN_loss_fake"] = gan_loss
    # Gradient penalty
    losses["gp_loss"] = gradient_regularization(
        grad_reg_tape, discriminator_output_real, real_latents
    )

    losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

    return losses


def gradient_regularization(grad_reg_tape, real_out, real_in):
    gradients_wrt_input = grad_reg_tape.gradient(real_out, real_in)
    gradients_sqr = tf.square(gradients_wrt_input)
    r1_penalty = tf.reduce_sum(gradients_sqr, axis=range(1, len(gradients_sqr.shape)))
    r1_penalty = tf.reduce_mean(r1_penalty)

    # weights from L. Mescheder et al.: Which Training Methods for GANs do actually Converge
    return 10 * 0.5 * r1_penalty


# Known as identity loss in HoloGAN
def compute_latent_regression_loss(generator_outputs, labels, latent_regressor):
    latent_regressor_output = latent_regressor(generator_outputs)
    latent_regression_loss = tf.losses.mean_squared_error(
        labels, latent_regressor_output
    )
    latent_regression_loss = tf.reduce_mean(latent_regression_loss)

    return latent_regression_loss
