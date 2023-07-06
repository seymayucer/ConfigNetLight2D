G_mb_ratio = 8 / (8 + 1)
D_mb_ratio = 16 / (16 + 1)


DEFAULT_CONFIG = {
    "config": "e",
    "impl": "cuda",
    "model_type": None,
    "latent_dim": 512,
    "output_shape": (256, 256, 3),
    "const_input_shape": (4, 4, 4, 512),
    "n_adain_mlp_layers": 2,
    "n_adain_mlp_units": 128,
    "gen_output_activation": "tanh",
    "n_discr_features_at_layer_0": 48,
    "max_discr_filters": 512,
    "n_discr_layers": 5,
    "discr_conv_kernel_size": 3,
    "use_style_discriminator": True,
    "rotation_ranges": ((-30, 30), (-10, 10), (0, 0)),
    "relu_before_in": True,
    "initial_from_rgb_layer_in_discr": True,
    # True gets better metrics but reduces stability
    "adain_on_learned_input": False,
    # Increasing improves rotation stability
    "latent_regressor_rot_weight": 5.0,
    "optimizer": {
        "learning_rate": 0.0004,
        "beta_1": 0.0,
        "beta_2": 0.9,
        "amsgrad": False,
    },
    "style_g_optimizer": {
        "learning_rate": 0.0025 * G_mb_ratio,
        "beta_1": 0.0 ** G_mb_ratio,
        "beta_2": 0.99 ** G_mb_ratio,
        "epsilon": 1e-8,
    },
    "style_d_optimizer": {
        "learning_rate": 0.0025 * D_mb_ratio,
        "beta_1": 0.0 ** G_mb_ratio,
        "beta_2": 0.99 ** D_mb_ratio,
        "epsilon": 1e-8,
    },
    "batch_size": 24,
    "latent_distribution": "normal",
    "metrics_checkpoint_period": 1000,
    "image_checkpoint_period": 1000,
    # Each input corresponds to a tuple where the first element is input dimensionality
    # the second element is corresponding dimensionality in latent space.
    # The first element should be filled by the process_metadata method of the training dataset.
    "facemodel_inputs": {
        "skin_color": (None, 3),
        "hair_color": (None, 3),
        "left_eye_features": (None, 125),
        "right_eye_features": (None, 125),
        "nose_features": (None, 128),
        "mouth_features": (None, 128),
    },
    "n_discriminator_updates": 1,
    "n_generator_updates": 1,
    "num_synth_encoder_layers": 2,
    "n_latent_discr_layers": 4,
    "image_loss_weight": 0.00005,
    "eye_loss_weight": 5,
    "domain_adverserial_loss_weight": 5.0,
    "latent_regression_weight": 0.0,
}
