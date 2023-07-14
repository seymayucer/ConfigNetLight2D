from collections import OrderedDict
import numpy as np
import sys
import os
import json
import pickle
import time
from tensorflow import keras
import tensorflow as tf
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from confignet.configs import DEFAULT_CONFIG
from confignet.perceptual_loss import PerceptualLoss
from confignet.focal_frequency_loss import FocalFrequencyLoss

from confignet.modules.synthetic_encoder import SyntheticDataEncoder
from confignet.neural_renderer_dataset import NeuralRendererDataset
from confignet import confignet_utils

from confignet.modules.stylegan2_discriminator import StyleGAN2Discriminator
from confignet.modules.stylegan2_generator import StyleGAN2Generator
from confignet.modules.building_blocks import MLPSimple
from confignet.metrics.metrics import InceptionMetrics

from confignet.metrics.metrics import ControllabilityMetrics
from confignet.losses import (
    compute_stylegan_discriminator_loss,
    compute_latent_discriminator_loss,
    compute_stylegan_generator_loss,
    GAN_G_loss,
    compute_latent_regression_loss,
)


class ConfigNetFirstStage:
    def __init__(self, config, initialize=True):
        self.config = confignet_utils.merge_configs(DEFAULT_CONFIG, config)

        self.config["model_type"] = "ConfigNetFirstStage"
        self.config["num_labels"] = 0

        self.generator = None
        self.generator_smoothed = None
        self.discriminator = None
        self.combined_model = None

        self.latent_discriminator = None

        self.log_writer = None
        self.g_losses = {}
        self.d_losses = {}
        self.metrics = {}

        self.latent_d_losses = {}

        self._checkpoint_visualization_input = None
        self._generator_input_for_metrics = None
        self._inception_metric_object = None

        self.n_checkpoint_samples = 10
        self.G_reg_interval = 8
        self.D_reg_interval = 16
        self.gamma = 0.0002 * (256**2) / self.config["batch_size"]

        # Remove the inputs that do not have a defined input dimensions
        self.config["facemodel_inputs"] = {
            key: value
            for key, value in self.config["facemodel_inputs"].items()
            if value[0] is not None
        }
        self.config["facemodel_inputs"] = OrderedDict(
            sorted(self.config["facemodel_inputs"].items(), key=lambda t: t[0])
        )

        self.config["latent_dim"] = 0
        for input_var_spec in self.config["facemodel_inputs"]:
            self.config["latent_dim"] += self.config["facemodel_inputs"][
                input_var_spec
            ][1]

        self.synthetic_encoder = None
        self.facemodel_param_distributions = None
        self.perceptual_loss = PerceptualLoss(
            self.config["output_shape"], model_type="imagenet"
        )
        self.focal_frequency_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)

        self.generator_fine_tuned = None
        self.controllability_metrics = None

        if initialize:
            self.initialize_network()

    def get_weights(self, return_tensors=False):
        weights = {}
        weights["generator_weights"] = self.generator.get_weights()
        weights["generator_smoothed_weights"] = self.generator_smoothed.get_weights()
        weights["discriminator_weights"] = self.discriminator.get_weights()
        weights["synthetic_encoder_weights"] = self.synthetic_encoder.get_weights()

        weights[
            "latent_discriminator_weights"
        ] = self.latent_discriminator.get_weights()

        return weights

    def set_weights(self, weights):
        self.generator.set_weights(weights["generator_weights"])
        self.generator_smoothed.set_weights(weights["generator_smoothed_weights"])
        self.discriminator.set_weights(weights["discriminator_weights"])
        self.synthetic_encoder.set_weights(weights["synthetic_encoder_weights"])
        self.latent_discriminator.set_weights(weights["latent_discriminator_weights"])

    def get_training_step_number(self):
        step_number = (
            0
            if "loss_sum" not in self.g_losses.keys()
            else len(self.g_losses["loss_sum"]) - 1
        )

        return step_number

    def get_batch_size(self):
        return self.config["batch_size"]

    def get_log_dict(self):
        log_dict = {
            "g_losses": self.g_losses,
            "d_losses": self.d_losses,
            "metrics": self.metrics,
        }

        return log_dict

    def set_logs(self, log_dict):
        self.g_losses = log_dict["g_losses"]
        self.d_losses = log_dict["d_losses"]
        self.metrics = log_dict["metrics"]

    def save(self, output_dir, output_filename):
        weights = self.get_weights()
        np.savez(os.path.join(output_dir, output_filename + ".npz"), **weights)
        with open(os.path.join(output_dir, output_filename + ".json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

        with open(
            os.path.join(output_dir, output_filename + "_facemodel_distr.pck"), "wb"
        ) as fp:
            pickle.dump(self.facemodel_param_distributions, fp)

    @classmethod
    def load(cls, file_path):
        print("First stage model is loading.")
        with open(file_path, "r") as fp:
            config = json.load(fp)

        model = cls(config)

        weigh_file = os.path.splitext(file_path)[0] + ".npz"
        weights = np.load(weigh_file, allow_pickle=True)
        model.set_weights(weights)

        log_file = os.path.splitext(file_path)[0] + "_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as fp:
                log_dict = json.load(fp)
            model.set_logs(log_dict)

        path_to_distribution_file = (
            os.path.splitext(file_path)[0] + "_facemodel_distr.pck"
        )
        if os.path.exists(path_to_distribution_file):
            with open(path_to_distribution_file, "rb") as fp:
                model.facemodel_param_distributions = pickle.load(fp)
        else:
            print("WARNING: facemodel param distributions not loaded")

        return model

    def initialize_network(self):
        # not trained
        self.synthetic_encoder = SyntheticDataEncoder(
            synthetic_encoder_inputs=self.config["facemodel_inputs"],
            num_layers=self.config["num_synth_encoder_layers"],
        )

        discriminiator_args = {
            "resolution": self.config["output_shape"][0],
            "config": self.config["config"],
            "impl": self.config["impl"],
            "mbstd_group_size": 4,
            "mbstd_num_features": 1,
            "name": "discriminator",
        }

        discriminator_input_shape = tuple(
            [self.config["batch_size"]] + list(self.config["output_shape"])
        )
        self.discriminator = StyleGAN2Discriminator(**discriminiator_args)
        self.discriminator.build(discriminator_input_shape)

        generator_args = {
            "resolution": self.config["output_shape"][0],
            "config": self.config["config"],
            "impl": self.config["impl"],
        }

        generator_input_shape = [[None, 512]]
        self.generator = StyleGAN2Generator(**generator_args)
        self.generator.build(generator_input_shape)

        generator_args["randomize_noise"] = False
        self.generator_smoothed = StyleGAN2Generator(**generator_args)
        self.generator_smoothed.build(generator_input_shape)
        self.generator_smoothed.set_weights(self.generator.get_weights())

        self.latent_discriminator = MLPSimple(
            num_layers=self.config["n_latent_discr_layers"],
            num_in=self.config["latent_dim"],
            num_hidden=self.config["latent_dim"],
            num_out=1,
            non_linear=keras.layers.LeakyReLU,
            non_linear_last=None,
        )
        # Used in identity loss, described in supplementary
        discriminiator_args = {
            "img_shape": self.config["output_shape"][:2],
            "num_resample": self.config["n_discr_layers"],
            "disc_kernel_size": self.config["discr_conv_kernel_size"],
            "disc_expansion_factor": self.config["n_discr_features_at_layer_0"],
            "disc_max_feature_maps": self.config["max_discr_filters"],
            "initial_from_rgb_layer_in_discr": self.config[
                "initial_from_rgb_layer_in_discr"
            ],
        }
   

    def synth_data_image_checkpoint(self, output_dir):
        step_number = self.get_training_step_number()

        facemodel_params = self._checkpoint_visualization_input["facemodel_params"]
        gt_imgs = self._checkpoint_visualization_input["gt_imgs"]

        generated_imgs = self.generate_images_from_facemodel(facemodel_params)
        generated_imgs = np.vstack((gt_imgs, generated_imgs))

        combined_image = confignet_utils.build_image_matrix(
            generated_imgs,
            generated_imgs.shape[0] // self.n_checkpoint_samples,
            self.n_checkpoint_samples,
        )

        img_output_dir = os.path.join(output_dir, "output_imgs")
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)

        cv2.imwrite(
            os.path.join(img_output_dir, str(step_number).zfill(6) + "_synth.jpg"),
            combined_image,
        )
        with self.log_writer.as_default():
            tf.summary.image(
                "generated_synth_images",
                combined_image[np.newaxis, :, :, [2, 1, 0]],
                step=step_number,
            )

    # Returns the total number of facemodel input dimensions
    @property
    def facemodel_input_dim(self):
        total_facemodel_input_dims = 0
        for facemodel_input_dim, _ in self.config["facemodel_inputs"].values():
            total_facemodel_input_dims += facemodel_input_dim

        return total_facemodel_input_dims

    def get_facemodel_param_idxs_in_latent(self, param_name):
        facemodel_param_dims = list(self.config["facemodel_inputs"].values())
        facemodel_param_names = list(self.config["facemodel_inputs"].keys())

        facemodel_param_idx = facemodel_param_names.index(param_name)

        start_idx = int(
            np.sum([x[1] for x in facemodel_param_dims[:facemodel_param_idx]])
        )
        end_idx = start_idx + facemodel_param_dims[facemodel_param_idx][1]

        return range(start_idx, end_idx)

    def set_facemodel_param_in_latents(self, latents, param_name, param_value):
        param_value = np.array(param_value)
        if len(param_value.shape) == 1:
            param_value = param_value[np.newaxis]
        latents_for_param = self.synthetic_encoder.per_facemodel_input_mlps[
            param_name
        ].predict(param_value)

        param_idxs_in_latent = self.get_facemodel_param_idxs_in_latent(param_name)

        new_latents = np.copy(latents)
        new_latents[:, param_idxs_in_latent] = latents_for_param

        return new_latents

    # Start of checkpoint-related code
    def image_checkpoint(self, output_dir):
        step_number = self.get_training_step_number()
        latent = self._checkpoint_visualization_input["latent"]

        # Predicted latent with predicted rotation
        generated_imgs = self.generate_images(latent)

        image_matrix = confignet_utils.build_image_matrix(
            generated_imgs,
            generated_imgs.shape[0] // self.n_checkpoint_samples,
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

        self.synth_data_image_checkpoint(output_dir)

    def sample_latent_vector(self, n_samples):
        if self.config["latent_distribution"] == "normal":
            # return np.random.normal(0, 1, (n_samples, self.config["latent_dim"]))
            return tf.random.normal(
                mean=0, stddev=1, shape=(n_samples, self.config["latent_dim"])
            )
        elif self.config["latent_distribution"] == "uniform":
            # return np.random.uniform(-1, 1, (n_samples, self.config["latent_dim"]))
            return tf.random.uniform(
                minval=-1, maxval=1, shape=(n_samples, self.config["latent_dim"])
            )

    def sample_synthetic_dataset(self, dataset, n_samples):
        sample_idxs = np.random.randint(0, dataset.imgs.shape[0], n_samples)

        facemodel_params = []
        for input_name in self.config["facemodel_inputs"].keys():
            facemodel_params.append(dataset.metadata_inputs[input_name][sample_idxs])
        gt_imgs = tf.convert_to_tensor(dataset.imgs[sample_idxs], dtype=tf.float32)
        # eye_masks = tf.convert_to_tensor(dataset.eye_masks[sample_idxs])

        return facemodel_params, gt_imgs  # , eye_masks

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
        real_imgs = (
            tf.convert_to_tensor(training_set.imgs[img_idxs], dtype=tf.float32) / 127.5
            - 1.0
        )
        real_imgs = tf.image.random_flip_left_right(real_imgs)
        latent_vector = self.sample_latent_vector(self.get_batch_size())
        fake_imgs = self.generator(latent_vector, training=True)
        return real_imgs, fake_imgs

    def get_synth_discriminator_batch(self, training_set):
        # Inputs
        img_idxs = np.random.randint(
            0, training_set.imgs.shape[0], self.get_batch_size()
        )
        real_imgs = (
            tf.convert_to_tensor(training_set.imgs[img_idxs], dtype=tf.float32) / 127.5
            - 1.0
        )
        real_imgs = tf.image.random_flip_left_right(real_imgs)

        facemodel_params, _ = self.sample_synthetic_dataset(
            training_set, self.get_batch_size()
        )
        latent_vector = self.synthetic_encoder(facemodel_params)

        fake_imgs = self.generator(latent_vector, training=True)

        return real_imgs, fake_imgs

    def update_smoothed_weights(self, smoother_alpha=0.999):
        training_weights = self.generator.get_weights()
        smoothed_weights = self.generator_smoothed.get_weights()

        for i in range(len(smoothed_weights)):
            smoothed_weights[i] = (
                smoother_alpha * smoothed_weights[i]
                + (1 - smoother_alpha) * training_weights[i]
            )

        self.generator_smoothed.set_weights(smoothed_weights)

    def generate_output_for_metrics(self):
        return self.generate_images(self._generator_input_for_metrics["latent"])

    # Start of evaluation code
    def calculate_metrics(self, output_dir):
        generated_images = self.generate_output_for_metrics()
        number_of_completed_iters = self.get_training_step_number()

        if "training_step_number" not in self.metrics.keys():
            self.metrics["training_step_number"] = []
        self.metrics["training_step_number"].append(number_of_completed_iters)
        self._inception_metric_object.update_and_log_metrics(
            generated_images, self.metrics, output_dir, self.log_writer
        )

    def run_checkpoints(self, output_dir, iteration_time, checkpoint_start=None):
        checkpoint_start = time.process_time()
        step_number = self.get_training_step_number()

        if step_number % self.config["image_checkpoint_period"] == 0:

            confignet_utils.log_loss_vals(
                self.latent_d_losses,
                output_dir,
                step_number,
                "latent_discriminator_",
                tb_log_writer=self.log_writer,
            )

        if checkpoint_start is None:
            checkpoint_start = time.process_time()()

        step_number = self.get_training_step_number()

        if step_number % self.config["metrics_checkpoint_period"] == 0:
            print("Running metrics")
            self.calculate_metrics(output_dir)
            checkpoint_output_dir = os.path.join(output_dir, "checkpoints")
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)
            print("Saving checkpoint")
            self.save(checkpoint_output_dir, "final")
            if step_number % 20000 == 0:
                self.save(checkpoint_output_dir, str(step_number).zfill(6))
            # # str(step_number).zfill(6))

        if step_number % self.config["image_checkpoint_period"] == 0:
            self.image_checkpoint(output_dir)
            confignet_utils.log_loss_vals(
                self.g_losses,
                output_dir,
                step_number,
                "generator_",
                tb_log_writer=self.log_writer,
            )
            confignet_utils.log_loss_vals(
                self.d_losses,
                output_dir,
                step_number,
                "discriminator_",
                tb_log_writer=self.log_writer,
            )

            # Only actually display the checkpoint time if the checkpoint is run
            checkpoint_end = time.process_time()

            checkpoint_time = checkpoint_end - checkpoint_start
            print("Training iteration time: %f" % (iteration_time))
            print("Checkpoint time: %f" % (checkpoint_time))

            with self.log_writer.as_default():
                tf.summary.scalar(
                    "perf/training_iter_time", iteration_time, step=step_number
                )
                tf.summary.scalar(
                    "perf/checkpoint_time", checkpoint_time, step=step_number
                )

    def discriminator_training_step(self, training_set, synth_training_set, optimizer):
        real_imgs, fake_imgs = self.get_discriminator_batch(training_set)
        syth_imgs, syth_fake_imgs = self.get_synth_discriminator_batch(
            synth_training_set
        )
        losses = {}
        with tf.GradientTape() as tape:
            if tf.math.equal(self.step % self.D_reg_interval, 0):
                """With r1 regulation."""
                real_losses, D_reg = compute_stylegan_discriminator_loss(
                    self.discriminator,
                    real_imgs,
                    fake_imgs,
                    gamma=self.gamma,
                    compute_reg=True,
                )
                synth_losses, D_reg = compute_stylegan_discriminator_loss(
                    self.discriminator,
                    syth_imgs,
                    syth_fake_imgs,
                    gamma=self.gamma,
                    compute_reg=True,
                )

                losses["real_d_loss"] = real_losses["loss_sum"]
                losses["synth_d_loss"] = synth_losses["loss_sum"]
                losses["loss_sum"] = (
                    0.5 * losses["real_d_loss"] + 0.5 * losses["synth_d_loss"]
                )
                losses["loss_sum"] += tf.reduce_mean(D_reg * self.D_reg_interval)
            else:
                real_losses, _ = compute_stylegan_discriminator_loss(
                    self.discriminator,
                    real_imgs,
                    fake_imgs,
                    gamma=self.gamma,
                    compute_reg=False,
                )
                synth_losses, _ = compute_stylegan_discriminator_loss(
                    self.discriminator,
                    syth_imgs,
                    syth_fake_imgs,
                    gamma=self.gamma,
                    compute_reg=False,
                )
                losses["real_d_loss"] = real_losses["loss_sum"]
                losses["synth_d_loss"] = synth_losses["loss_sum"]
                losses["loss_sum"] = (
                    0.5 * losses["real_d_loss"] + 0.5 * losses["synth_d_loss"]
                )
        if losses["loss_sum"] is None:
            print("discriminator_training_step")
            breakpoint()

        trainable_weights = self.discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))
        return losses

    def latent_discriminator_training_step(self, synth_training_set, optimizer):
        # Inputs
        real_latents = self.sample_latent_vector(self.get_batch_size())
        facemodel_params, _ = self.sample_synthetic_dataset(
            synth_training_set, self.get_batch_size()
        )
        fake_latents = self.synthetic_encoder(facemodel_params)

        with tf.GradientTape() as tape:
            losses = compute_latent_discriminator_loss(
                self.latent_discriminator, real_latents, fake_latents
            )
        if losses["loss_sum"] is None:
            print("latent_discriminator_training_step")
            breakpoint()
        trainable_weights = self.latent_discriminator.trainable_weights
        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def generator_training_step(self, real_training_set, synth_training_set, optimizer):
        n_synth_in_batch = self.get_batch_size() // 2
        n_real_in_batch = self.get_batch_size() - n_synth_in_batch

        # Synth batch
        (facemodel_params, gt_imgs) = self.sample_synthetic_dataset(
            synth_training_set, n_synth_in_batch
        )

        gt_imgs = gt_imgs / 127.5 - 1.0
        # Real batch
        real_latents = self.sample_latent_vector(n_real_in_batch)

        # # Labels for gan loss
        # valid_y_synth = tf.ones((n_synth_in_batch, 1))
        # fake_y_real = tf.zeros((n_real_in_batch, 1))

        # domain_adverserial_loss_labels = tf.experimental.numpy.vstack(
        #     (fake_y_real, valid_y_synth)
        # )

        losses = {}
        # Generator Step

        with tf.GradientTape() as tape:
            synth_latents = self.synthetic_encoder(facemodel_params)

            generator_output_synth = self.generator(synth_latents)
            generator_output_real = self.generator(real_latents)

            # image loss 0
          
            losses["image_loss"] = self.config[
                "image_loss_weight"
            ] * self.perceptual_loss.loss(gt_imgs, generator_output_synth)
            # losses["focal"] = 1 * self.focal_frequency_loss(
            #     tf.transpose(gt_imgs, perm=[0, 3, 1, 2]),
            #     tf.transpose(generator_output_synth, perm=[0, 3, 1, 2]),
            # )
        

            discriminator_output_synth, synth_latents_pred = self.discriminator(
                generator_output_synth,
                training=True,
            )

            # GAN loss for real
            discriminator_output_real, real_latents_pred = self.discriminator(
                generator_output_real,
                training=True,
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

                # latent_gan_loss, latent_gan_reg = compute_stylegan_generator_loss(
                #     self.generator, latent_discriminator_output
                # )
                # latent_gan_loss += tf.reduce_mean(latent_gan_reg * self.G_reg_interval)

            else:
                G_loss_real, _ = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_real
                )
                G_loss_synth, _ = compute_stylegan_generator_loss(
                    self.generator, discriminator_output_synth
                )
                # latent_gan_loss, _ = compute_stylegan_generator_loss(
                #     self.generator, latent_discriminator_output
                # )
            if G_loss_real is None or G_loss_synth is None:
                print("generator_training_step")
                breakpoint()
            losses["GAN_loss_real"] = G_loss_real
            losses["GAN_loss_synth"] = G_loss_synth

        
            # Domain adverserial loss
            latent_discriminator_output = self.latent_discriminator(synth_latents)
            latent_gan_loss = GAN_G_loss(latent_discriminator_output)
            losses["latent_GAN_loss"] = (
                self.config["domain_adverserial_loss_weight"] * latent_gan_loss
            )
            # Latent regression loss start
          
            # identity loss it is not workking well
            # losses["latent_regression_loss"] = self.config[
            #     "latent_regression_weight"
            # ] * tf.reduce_mean(tf.square(synth_latents - synth_latents_pred))
                
                
            losses["loss_sum"] = tf.reduce_sum(list(losses.values()))

        trainable_weights = (
            self.generator.trainable_weights
            + self.synthetic_encoder.trainable_weights
        )

        gradients = tape.gradient(losses["loss_sum"], trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return losses

    def setup_training(
        self,
        log_dir,
        synth_training_set,
        n_samples_for_metrics,
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

    def train(
        self,
        real_training_set,
        synth_training_set,
        validation_set,
        output_dir,
        log_dir,
        n_steps=100000,
        n_samples_for_metrics=1000,
    ):
        self.setup_training(
            log_dir,
            synth_training_set,
            n_samples_for_metrics,
            real_training_set=real_training_set,
            validation_set=validation_set,
        )
        start_step = self.get_training_step_number()
        print(f"Starting training at step {start_step}")

        discriminator_optimizer = keras.optimizers.Adam(
            **self.config["style_d_optimizer"]
        )
        generator_optimizer = keras.optimizers.Adam(**self.config["style_g_optimizer"])

        for step in range(start_step, n_steps):
            training_iteration_start = time.process_time()

            self.step = step
            for _ in range(self.config["n_discriminator_updates"]):
                d_loss = self.discriminator_training_step(
                    real_training_set, synth_training_set, discriminator_optimizer
                )
                latent_d_loss = self.latent_discriminator_training_step(
                    synth_training_set, discriminator_optimizer
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
                    d_loss["real_d_loss"],
                    d_loss["synth_d_loss"],
                    latent_d_loss["loss_sum"],
                    g_loss["loss_sum"],
                )
            )
            confignet_utils.update_loss_dict(self.g_losses, g_loss)
            confignet_utils.update_loss_dict(self.d_losses, d_loss)
            confignet_utils.update_loss_dict(self.latent_d_losses, latent_d_loss)

            iteration_time = training_iteration_end - training_iteration_start
            self.run_checkpoints(output_dir, iteration_time)

    def generate_images(self, latent_vector):
        imgs = self.generator_smoothed.predict(latent_vector)
        imgs = np.clip(imgs, -1.0, 1.0)
        imgs = ((imgs + 1) * 127.5).astype(np.uint8)

        return imgs

    def generate_images_from_facemodel(self, facemodel_params):
        latent_vectors = self.synthetic_encoder(facemodel_params)
        return self.generate_images([latent_vectors])

    def fit_facemodel_expression_params_to_latent(
        self,
        latent,
        unused_expr_idxs=None,
        param_name="blendshape_values",
        n_iters=2000,
        learning_rate=0.05,
        verbose=False,
    ):
        expression_idxs_in_latent = self.get_facemodel_param_idxs_in_latent(param_name)

        facemodel_param_names = list(self.config["facemodel_inputs"].keys())
        facemodel_param_dims = list(self.config["facemodel_inputs"].values())
        expression_idx_in_facemodel_params = list(facemodel_param_names).index(
            param_name
        )

        latent_exp_values = latent[:, expression_idxs_in_latent]
        facemodel_param_values = tf.zeros(
            (1, facemodel_param_dims[expression_idx_in_facemodel_params][0]),
            dtype=tf.float32,
        )
        facemodel_param_values = tf.Variable(facemodel_param_values)

        synthetic_encoder_model = self.synthetic_encoder.per_facemodel_input_mlps[
            param_name
        ]

        optimizer = keras.optimizers.SGD(lr=learning_rate)

        for step in range(n_iters):
            with tf.GradientTape() as tape:
                predicted_latent = synthetic_encoder_model(facemodel_param_values)

                loss = tf.reduce_mean(tf.square(latent_exp_values - predicted_latent))

            gradients = tape.gradient(loss, [facemodel_param_values])
            optimizer.apply_gradients(zip(gradients, [facemodel_param_values]))
            facemodel_param_values.assign(
                tf.clip_by_value(facemodel_param_values, 0, 1)
            )

            if unused_expr_idxs is not None:
                facemodel_param_values_numpy = facemodel_param_values.numpy()
                facemodel_param_values_numpy[:, unused_expr_idxs] = 0
                facemodel_param_values.assign(facemodel_param_values_numpy)

            if verbose:
                print("%d: %f" % (step, loss.numpy()))

        return facemodel_param_values.numpy()
