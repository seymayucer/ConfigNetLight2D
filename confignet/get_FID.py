import cv2
import numpy as np
import os
import sys
from pathlib import Path

from metrics.inception_distance import (
    InceptionFeatureExtractor,
    compute_FID,
    compute_KID,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from confignet_second_stage import ConfigNet


class FIDTest:
    def __init__(
        self, test_image_path, confignet_model_path, latent_gan_model_path=None
    ) -> None:
        # if self.latentgan_model is not None:
        #     self.latentgan_model = LatentGAN.load(latent_gan_model_path)
        print("Loading ConfigNet model...")
        self.confignet_model = ConfigNet.load(confignet_model_path)
        self.test_image_path = test_image_path
        self.test_imgs_list = list(sorted(Path(self.test_image_path).glob("*.png")))
        self.inception_feature_extractor = InceptionFeatureExtractor((256, 256, 3))

    def get_batch_images(self, batch_images):
        images = []

        for image in batch_images:
            image = cv2.imread(str(image))
            images.append(image)

        return ((np.asanyarray(images) + 1) * 127.5).astype(np.uint8)

    def get_fid(self, n_images, batch_size):
        # keep track of average FID score
        mean_fid = 0
        for i in range(0, n_images, batch_size):
            batch_images = self.test_imgs_list[i : i + batch_size]
            gt_images = self.get_batch_images(batch_images)
            latents = self.confignet_model.encode_images(gt_images)
            generated = self.confignet_model.generate_images(latents)
            self.confignet_model.fine_tune_on_imgs(gt_images, 1)

            generated_inception_features = (
                self.inception_feature_extractor.get_features(generated)
            )
            gt_inception_features = self.inception_feature_extractor.get_features(
                gt_images
            )

            current_fid = compute_FID(
                generated_inception_features, gt_inception_features
            )
            mean_fid += current_fid
            print("FID score for batch", i, "-", i + batch_size, ":", current_fid)

        mean_fid /= n_images / batch_size
        return mean_fid


fid_object = FIDTest(
    test_image_path="/home2/xcnf86/confignet_stylegan/FFHQ_test",
    confignet_model_path="/home2/xcnf86/confignet_stylegan/ConfigNetLight2D/experiments/celeba_focalin1st_single_discriminator_2g1d/checkpoints/final.json",
    latent_gan_model_path=None,
)

fid_object.get_fid(n_images=10000, batch_size=10000)


# inception_feature_extractor = InceptionFeatureExtractor((256, 256, 3))
# generated_image = cv2.imread('/mnt/SSD/confignet_stylegan/ConfigNetLight2D/samples/my_people/generated/TB.png')
# gt_image = cv2.imread('/mnt/SSD/confignet_stylegan/ConfigNetLight2D/samples/my_people/normalized/TB.png')

# generated_inception_features = inception_feature_extractor.get_features(generated_image[np.newaxis])
# gt_inception_features= inception_feature_extractor.get_features(gt_image[np.newaxis])

# #duplicate the arrays by adding same element to the array
# generated_inception_features = np.concatenate([generated_inception_features, generated_inception_features], axis=0)
# gt_inception_features = np.concatenate([gt_inception_features, gt_inception_features], axis=0)
# print(gt_inception_features.shape,generated_inception_features.shape)
# #convert tf tensor to numpy matrix

# fid = compute_FID(generated_inception_features, gt_inception_features)
# print('FID score between generated images',fid)
