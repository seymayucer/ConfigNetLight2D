from collections import OrderedDict
import numpy as np
import sys
import os
import time
from tensorflow import keras
import tensorflow as tf
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from confignet.configs import DEFAULT_CONFIG
from confignet.perceptual_loss import PerceptualLoss
from confignet.modules.synthetic_encoder import SyntheticDataEncoder
from confignet.neural_renderer_dataset import NeuralRendererDataset
from confignet import confignet_utils
from confignet.modules.real_encoder import RealEncoder
from confignet.modules.stylegan2_discriminator import StyleGAN2Discriminator
from confignet.modules.stylegan2_generator import StyleGAN2Generator
from confignet.modules.building_blocks import MLPSimple
from confignet.metrics.metrics import InceptionMetrics
from confignet.modules.hologan_discriminator import HologanLatentRegressor
from confignet.metrics.metrics import ControllabilityMetrics
from confignet.losses import (
    compute_stylegan_discriminator_loss,
    compute_latent_discriminator_loss,
    compute_stylegan_generator_loss,
    GAN_D_loss,
    eye_loss,
)

from .confignet_first_stage import ConfigNetFirstStage


class ConfigNet(ConfigNetFirstStage):
    def __init__(self, config, initialize=True):
        self.config = confignet_utils.merge_configs(DEFAULT_CONFIG, config)

        super(ConfigNet, self).__init__(self.config, initialize=False)
        self.config["model_type"] = "ConfigNet"

        self.encoder = None
        self.generator_fine_tuned = None
        self.controllability_metrics = None
        self.perceptual_loss_face_reco = PerceptualLoss(
            self.config["output_shape"], model_type="VGGFace"
        )

        if initialize:
            self.initialize_network()

    def get_weights(self):
        weights = super().get_weights()
        weights["real_encoder_weights"] = self.encoder.get_weights()

        return weights

    def set_weights(self, weights):
        super().set_weights(weights)
        self.encoder.set_weights(weights["real_encoder_weights"])

    def initialize_network(self):
        super(ConfigNet, self).initialize_network()

        self.encoder = RealEncoder(
            self.config["latent_dim"], self.config["output_shape"],
        )
        self.encoder(np.zeros((1, *self.config["output_shape"]), np.float32))

        # # Used in identity loss, described in supplementary
        # discriminiator_args = {
        #     "img_shape": self.config["output_shape"][:2],
        #     "num_resample": self.config["n_discr_layers"],
        #     "disc_kernel_size": self.config["discr_conv_kernel_size"],
        #     "disc_expansion_factor": self.config["n_discr_features_at_layer_0"],
        #     "disc_max_feature_maps": self.config["max_discr_filters"],
        #     "initial_from_rgb_layer_in_discr": self.config[
        #         "initial_from_rgb_layer_in_discr"
        #     ],
        # }
        # discriminator_input_shape = tuple(
        #     [self.config["batch_size"]] + list(self.config["output_shape"])
        # )
        # self.latent_regressor = HologanLatentRegressor(
        #     self.config["latent_dim"], **discriminiator_args
        # )
        # self.latent_regressor.build(discriminator_input_shape)

    def image_checkpoint(self, output_dir):
        self.synth_data_image_checkpoint(output_dir)

        # Autoencoder checkpoint start
        step_number = self.get_training_step_number()

        gt_imgs = self._checkpoint_visualization_input["input_images"]
        latent = self.encode_images(gt_imgs)

        # Predicted latent with predicted rotation
        generated_images = self.generate_images(latent)
        parameter_name = "skin_color"
        skin_color_latents = latent
        gt_imgs_0_255 = ((gt_imgs.numpy() + 1) * 127.5).astype(np.uint8)
        combined_images = np.vstack((gt_imgs_0_255, generated_images))
        for idns in range(5):
            new_param_value = self.facemodel_param_distributions[parameter_name].sample(
                1
            )[0]

            skin_color_latents = self.set_facemodel_param_in_latents(
                skin_color_latents, parameter_name, new_param_value
            )

            generated_skin_images = self.generate_images(skin_color_latents)
            combined_images = np.vstack((combined_images, generated_skin_images))

        image_matrix = confignet_utils.build_image_matrix(
            combined_images,
            combined_images.shape[0] // self.n_checkpoint_samples,
            self.n_checkpoint_samples,
        )

        img_output_dir = os.path.join(output_dir, "output_imgs")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        image_grid_file_path = os.path.join(
            img_output_dir, str(step_number).zfill(6) + ".png"
        )

        cv2.imwrite(image_grid_file_path, image_matrix)

        with self.log_writer.as_default():
            tf.summary.image(
                "generated_images",
                image_matrix[np.newaxis, :, :, [2, 1, 0]],
                step=step_number,
            )

    def generate_output_for_metrics(self):
        latent = self.encode_images(self._generator_input_for_metrics["input_images"])
        return self.generate_images([latent])

    # Start of training code
    def face_reco_loss(self, gt_imgs, gen_imgs):
        loss_vals = self.perceptual_loss_face_reco.loss(gen_imgs, gt_imgs)

        return tf.reduce_mean(loss_vals)

    def compute_normalized_latent_regression_loss(self, generator_outputs, labels):
        latent_regressor_output = self.latent_regressor(generator_outputs)

        denominator = tf.sqrt(
            tf.math.reduce_variance(labels, axis=0, keepdims=True) + 1e-3
        )
        # Do not normalize the rotation element
        # denominator = tf.concat((denominator[:, :-3], tf.ones((1, 3), tf.float32)), axis=1)

        latent_regressor_output = (
            tf.reduce_mean(latent_regressor_output, axis=0)
            + (
                latent_regressor_output
                - tf.reduce_mean(latent_regressor_output, axis=0)
            )
            / denominator
        )
        labels = (
            tf.reduce_mean(labels, axis=0)
            + (labels - tf.reduce_mean(labels, axis=0)) / denominator
        )

        latent_regression_loss = tf.losses.mean_squared_error(
            labels, latent_regressor_output
        )
        latent_regression_loss = tf.reduce_mean(latent_regression_loss)
        latent_regression_loss *= self.config["latent_regression_weight"]

        return latent_regression_loss

    def sample_random_batch_of_images(self, dataset, batch_size=None):
        if batch_size is None:
            batch_size = self.get_batch_size()
        img_idxs = np.random.randint(0, dataset.imgs.shape[0], batch_size)
        imgs = (
            tf.convert_to_tensor(dataset.imgs[img_idxs], dtype=tf.float32) / 127.5 - 1.0
        )

        imgs = tf.image.random_flip_left_right(imgs)

        return imgs

    def get_discriminator_batch(self, training_set):
        # Inputs
        img_idxs = np.random.randint(
            0, training_set.imgs.shape[0], self.get_batch_size()
        )
        real_imgs = tf.convert_to_tensor(training_set.imgs[img_idxs], dtype=tf.float32)
        real_imgs = real_imgs / 127.5 - 1.0

        real_imgs = tf.image.flip_left_right(real_imgs)
        latent_vector = self.sample_latent_vector(self.get_batch_size())

        fake_imgs = self.generator(latent_vector, training=True)

        return real_imgs, fake_imgs

    def latent_discriminator_training_step(
        self, real_training_set, synth_training_set, optimizer
    ):
        # Inputs
        real_imgs = self.sample_random_batch_of_images(real_training_set)
        real_latents = self.encoder(real_imgs)
        facemodel_params, _ = self.sample_synthetic_dataset(
            synth_training_set, self.get_batch_size()
        )
        fake_latents = self.synthetic_encoder(facemodel_params)

        with tf.GradientTape() as tape:
            losses = compute_latent_discriminator_loss(
                self.latent_discriminator, real_latents, fake_latents
            )

        trainable_weights = self.latent_discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def generator_training_step(self, real_training_set, synth_training_set, optimizer):
        n_synth_in_batch = self.get_batch_size() // 2
        n_real_in_batch = self.get_batch_size() - n_synth_in_batch

        # Synth batch
        (facemodel_params, synth_imgs) = self.sample_synthetic_dataset(
            synth_training_set, n_synth_in_batch
        )
        # synth_imgs = synth_imgs.astype(np.float32) / 127.5 - 1.0
        synth_imgs = synth_imgs / 127.5 - 1.0
        # Real batch
        real_imgs = self.sample_random_batch_of_images(
            real_training_set, n_real_in_batch
        )

        # Labels for gan loss
        valid_y_synth = tf.ones((n_synth_in_batch, 1))
        fake_y_real = tf.zeros((n_real_in_batch, 1))

        domain_adverserial_loss_labels = tf.experimental.numpy.vstack(
            (fake_y_real, valid_y_synth)
        )

        losses = {}
        # Generator Step
        with tf.GradientTape() as tape:
            tape.watch([real_imgs, synth_imgs])
            synth_latents = self.synthetic_encoder(facemodel_params)
            generator_output_synth = self.generator(synth_latents)

            real_latents = self.encoder(real_imgs)
            generator_output_real = self.generator(real_latents)

            losses["image_loss_synth"] = self.config[
                "image_loss_weight"
            ] * self.perceptual_loss.loss(synth_imgs, generator_output_synth)
            losses["image_loss_real"] = self.config[
                "image_loss_weight"
            ] * self.perceptual_loss.loss(real_imgs, generator_output_real)
            # losses["eye_loss"] = self.config["eye_loss_weight"] * eye_loss(
            #     synth_imgs, generator_output_synth, eye_masks
            # )
            losses["focal_synth"] = 1 * self.focal_frequency_loss(
                    tf.transpose(synth_imgs, perm=[0, 3, 1, 2]),
                    tf.transpose(generator_output_synth, perm=[0, 3, 1, 2]),
                )
            losses["focal_real"] = 1 * self.focal_frequency_loss(
                    tf.transpose(real_imgs, perm=[0, 3, 1, 2]),
                    tf.transpose(generator_output_real, perm=[0, 3, 1, 2]),
                )
            
            # GAN loss for synth
            discriminator_output_synth, _ = self.synth_discriminator(
                generator_output_synth, training=True,
            )
            # GAN loss for real
            discriminator_output_real, _ = self.discriminator(
                generator_output_real, training=True,
            )

            if tf.math.equal(self.step % self.D_reg_interval, 0):
                """With pl regulation."""
                G_loss_real, G_reg_real = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_real
                )
                G_loss_real += tf.reduce_mean(G_reg_real * self.G_reg_interval)

                # GAN loss for synth
                G_loss_synth, G_reg_synth = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_synth
                )
                G_loss_synth += tf.reduce_mean(G_reg_synth * self.G_reg_interval)

            else:
                G_loss_real, _ = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_real
                )
                G_loss_synth, _ = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_synth
                )

            losses["GAN_loss_real"] = G_loss_real
            losses["GAN_loss_synth"] = G_loss_synth

            # Domain adverserial loss
            latent_discriminator_out_synth = self.latent_discriminator(synth_latents)
            latent_discriminator_out_real = self.latent_discriminator(real_latents)

            latent_discriminator_output = tf.concat(
                (latent_discriminator_out_real, latent_discriminator_out_synth), axis=0
            )

            latent_gan_loss = GAN_D_loss(
                domain_adverserial_loss_labels, latent_discriminator_output
            )

            if (
                tf.math.is_nan(G_loss_real)
                or tf.math.is_nan(G_loss_synth)
                or tf.math.is_nan(latent_gan_loss)
            ):
                print("second stage generator_training_step")
                breakpoint()
            losses["latent_GAN_loss"] = (
                self.config["domain_adverserial_loss_weight"] * latent_gan_loss
            )

            # if self.config["latent_regression_weight"] > 0.0:
            # Latent regression loss start
            # stacked_latent_vectors = tf.concat(
            #     (synth_latents, real_latents), axis=0
            # )
            # stacked_generated_imgs = tf.concat(
            #     (generator_output_synth, generator_output_real), axis=0
            # )

            # losses[
            #     "latent_regression_loss"
            # ] = self.compute_normalized_latent_regression_loss(
            #     stacked_generated_imgs, stacked_latent_vectors
            # )

            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = (
            self.generator.trainable_weights
            # + self.latent_regressor.trainable_weights
            + self.synthetic_encoder.trainable_weights
        )
        trainable_weights += self.encoder.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def calculate_metrics(self, output_dir):
        generated_images = self.generate_output_for_metrics()
        number_of_completed_iters = self.get_training_step_number()

        if "training_step_number" not in self.metrics.keys():
            self.metrics["training_step_number"] = []
        self.metrics["training_step_number"].append(number_of_completed_iters)
        self._inception_metric_object.update_and_log_metrics(
            generated_images, self.metrics, output_dir, self.log_writer
        )

    def setup_training(
        self,
        log_dir,
        synth_training_set,
        n_samples_for_metrics,
        attribute_classifier,
        real_training_set,
        validation_set,
    ):
        if real_training_set is None:
            real_training_set = synth_training_set

        os.makedirs(log_dir, exist_ok=True)
        self.log_writer = tf.summary.create_file_writer(log_dir)

        self._inception_metric_object = InceptionMetrics(self.config, real_training_set)

        self._generator_input_for_metrics = {}
        self._generator_input_for_metrics["latent"] = self.sample_latent_vector(
            n_samples_for_metrics
        )

        checkpoint_latent = self.sample_latent_vector(self.n_checkpoint_samples)

        self._checkpoint_visualization_input = {}
        self._checkpoint_visualization_input["latent"] = checkpoint_latent

        self.facemodel_param_distributions = (
            synth_training_set.metadata_input_distributions
        )

        facemodel_params, gt_imgs = self.sample_synthetic_dataset(
            synth_training_set, self.n_checkpoint_samples
        )

        for i, param in enumerate(facemodel_params):
            facemodel_params[i] = np.tile(param, (1, 1))

        self._checkpoint_visualization_input["facemodel_params"] = facemodel_params
        self._checkpoint_visualization_input["gt_imgs"] = gt_imgs

        sample_idxs = np.random.randint(
            0, validation_set.imgs.shape[0], self.n_checkpoint_samples
        )
        # checkpoint_input_imgs = validation_set.imgs[sample_idxs].astype(np.float32)
        checkpoint_input_imgs = tf.convert_to_tensor(
            validation_set.imgs[sample_idxs], dtype=tf.float32
        )
        self._checkpoint_visualization_input["input_images"] = (
            checkpoint_input_imgs / 127.5
        ) - 1.0

        sample_idxs = np.random.randint(
            0, validation_set.imgs.shape[0], n_samples_for_metrics
        )
        metric_input_imgs = tf.convert_to_tensor(
            validation_set.imgs[sample_idxs], dtype=tf.float32
        )
        self._generator_input_for_metrics["input_images"] = (
            metric_input_imgs / 127.5
        ) - 1.0

        self.controllability_metrics = ControllabilityMetrics(
            self, attribute_classifier
        )

    def train(
        self,
        real_training_set,
        synth_training_set,
        validation_set,
        attribute_classifier,
        output_dir,
        log_dir,
        n_steps=100000,
        n_samples_for_metrics=1000,
    ):
        self.setup_training(
            log_dir,
            synth_training_set,
            n_samples_for_metrics,
            attribute_classifier,
            real_training_set=real_training_set,
            validation_set=validation_set,
        )
        start_step = self.get_training_step_number()
        print(f"Starting training at step {start_step}")

        discriminator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])
        generator_optimizer = keras.optimizers.Adam(**self.config["optimizer"])

        for step in range(start_step, n_steps):
            training_iteration_start = time.process_time()

            for _ in range(self.config["n_discriminator_updates"]):
                self.step = step
                d_loss = self.discriminator_training_step(
                    real_training_set, discriminator_optimizer
                )

                synth_d_loss = self.synth_discriminator_training_step(
                    synth_training_set, discriminator_optimizer
                )

                latent_d_loss = self.latent_discriminator_training_step(
                    real_training_set, synth_training_set, discriminator_optimizer
                )

            for _ in range(self.config["n_generator_updates"]):
                g_loss = self.generator_training_step(
                    real_training_set, synth_training_set, generator_optimizer
                )

            self.update_smoothed_weights()

            training_iteration_end = time.process_time()
            print(
                "[D loss: %f] [synth_D loss: %f] [latent_D_loss: %f] [G loss: %f]"
                % (
                    d_loss["loss_sum"],
                    synth_d_loss["loss_sum"],
                    latent_d_loss["loss_sum"],
                    g_loss["loss_sum"],
                )
            )
            confignet_utils.update_loss_dict(self.g_losses, g_loss)
            confignet_utils.update_loss_dict(self.d_losses, d_loss)
            confignet_utils.update_loss_dict(self.synth_d_losses, synth_d_loss)
            confignet_utils.update_loss_dict(self.latent_d_losses, latent_d_loss)

            iteration_time = training_iteration_end - training_iteration_start
            self.run_checkpoints(output_dir, iteration_time)

    def encode_images(self, input_images):
        if input_images.dtype == np.uint8 or input_images.dtype == tf.uint8:
            input_images = (
                tf.convert_to_tensor(input_images, dtype=tf.float32) / 127.5 - 1.0
            )
        embeddings = self.encoder.predict(input_images)

        return embeddings

    def generate_images(self, latent_vector):
        imgs = self.generator_smoothed.predict(latent_vector)
        imgs = np.clip(imgs, -1.0, 1.0)
        imgs = ((imgs + 1) * 127.5).astype(np.uint8)

        return imgs

    def fine_tune_on_img(
        self,
        input_images,
        n_iters=50,
        img_output_dir=None,
        force_neutral_expression=False,
    ):
        if input_images.dtype == np.uint8:
            input_images = (input_images / 127.5) - 1.0
        if len(input_images.shape) == 3:
            input_images = input_images[np.newaxis]

        predicted_embeddings, predicted_rotations = self.encoder.predict(input_images)
        if force_neutral_expression:
            n_exp_blendshapes = self.config["facemodel_inputs"]["blendshape_values"][0]
            neutral_expr_params = np.zeros((1, n_exp_blendshapes), np.float32)
            predicted_embeddings = self.set_facemodel_param_in_latents(
                predicted_embeddings, "blendshape_values", neutral_expr_params
            )

        if self.generator_fine_tuned is None:
            self.generator_fine_tuned = StyleGAN2Generator(
                **self._get_generator_kwargs()
            )
            # Run once to generate weights
            self.generator_fine_tuned(predicted_embeddings)
        self.generator_fine_tuned.set_weights(self.generator_smoothed.get_weights())

        expr_idxs = self.get_facemodel_param_idxs_in_latent("blendshape_values")
        mean_predicted_embedding = np.mean(predicted_embeddings, axis=0, keepdims=True)

        pre_expr_embeddings = tf.Variable(mean_predicted_embedding[:, : expr_idxs[0]])
        expr_embeddings = tf.Variable(predicted_embeddings[:, expr_idxs])
        post_expr_embeddings = tf.Variable(
            mean_predicted_embedding[:, expr_idxs[-1] + 1 :]
        )
        n_imgs = input_images.shape[0]

        optimizer = keras.optimizers.Adam(lr=0.0001)
        fake_y_real = np.ones((1, 1))

        convert_to_uint8 = lambda x: ((x[0] + 1) * 127.5).astype(np.uint8)

        if img_output_dir is not None:
            os.makedirs(img_output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(img_output_dir, "gt_img.png"),
                convert_to_uint8(input_images),
            )

        for step_number in range(n_iters):
            losses = {}

            with tf.GradientTape() as tape:
                pre_expr_embeddings_tiled = tf.tile(pre_expr_embeddings, (n_imgs, 1))
                post_expr_embeddings_tiled = tf.tile(post_expr_embeddings, (n_imgs, 1))

                embeddings = tf.concat(
                    (
                        pre_expr_embeddings_tiled,
                        expr_embeddings,
                        post_expr_embeddings_tiled,
                    ),
                    axis=1,
                )

                generator_output_real = self.generator_fine_tuned(embeddings)
                losses["image_loss_real"] = (
                    0.5
                    * self.config["image_loss_weight"]
                    * self.perceptual_loss.loss(input_images, generator_output_real)
                )
                losses["face_reco_loss"] = (
                    0.5
                    * self.config["image_loss_weight"]
                    * self.face_reco_loss(input_images, generator_output_real)
                )

                # GAN loss for real
                discriminator_output_real = self.discriminator(generator_output_real)
                for i, disc_out in enumerate(discriminator_output_real.values()):
                    gan_loss = GAN_G_loss(disc_out)
                    losses["GAN_loss_real_" + str(i)] = gan_loss

                # Domain adverserial loss
                latent_discriminator_out_real = self.latent_discriminator(embeddings)
                latent_gan_loss = GAN_D_loss(fake_y_real, latent_discriminator_out_real)

                losses["latent_GAN_loss"] = (
                    self.config["domain_adverserial_loss_weight"] * latent_gan_loss
                )

                # Latent regression loss start
                latent_regression_labels = tf.concat(
                    (embeddings, self.config["latent_regressor_rot_weight"],), axis=-1,
                )

                # Regression of Z and rotation from output image
                # losses[
                #     "latent_regression_loss"
                # ] = self.compute_normalized_latent_regression_loss(
                #     generator_output_real, latent_regression_labels
                # )

                losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

            trainable_weights = self.generator_fine_tuned.trainable_weights + [
                pre_expr_embeddings,
                post_expr_embeddings,
            ]
            if not force_neutral_expression:
                trainable_weights.append(expr_embeddings)
            gradients = tape.gradient(losses["loss_sum"], trainable_weights)
            optimizer.apply_gradients(zip(gradients, trainable_weights))

            print(losses["loss_sum"])
            if img_output_dir is not None:
                cv2.imwrite(
                    os.path.join(img_output_dir, "output_%02d.png" % (step_number)),
                    convert_to_uint8(generator_output_real.numpy()),
                )

        embeddings = tf.concat(
            (pre_expr_embeddings_tiled, expr_embeddings, post_expr_embeddings_tiled),
            axis=1,
        )
        return embeddings.numpy()

