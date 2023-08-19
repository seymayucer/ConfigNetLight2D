import cv2
import numpy as np
import os
import sys
from pathlib import Path
import tqdm
import tensorflow as tf
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from confignet.confignet_second_stage import ConfigNet
from confignet.confignet_first_stage import ConfigNetFirstStage
from confignet.latent_gan import LatentGAN
from confignet.metrics.inception_distance import (
    InceptionFeatureExtractor,
    compute_FID,
)


class FIDTest:
    def __init__(
        self,
        test_image_path,
        confignet_model_path,
        first_stage_model_path=None,
        latent_gan_model_path=None,
    ) -> None:
        self.test_output = Path(confignet_model_path).parent.parent
        self.file = open(self.test_output / "FID.txt", "w+")

        # FFHQ Test 10k
        self.test_image_path = test_image_path
        self.test_imgs_list = list(sorted(Path(self.test_image_path).glob("*.png")))

        self.inception_feature_extractor = InceptionFeatureExtractor((299, 299, 3))

        print("Loading LatentGAN model...")
        if latent_gan_model_path is not None:
            self.latentgan_model = LatentGAN.load(latent_gan_model_path)

        print("Loading First Stage ConfigNet model...")
        if first_stage_model_path is not None:
            self.first_stage_model = ConfigNetFirstStage.load(first_stage_model_path)

        print("Loading ConfigNet model...")
        if confignet_model_path is not None:
            self.confignet_model = ConfigNet.load(confignet_model_path)

        # rgb bgr
        self.idx = 0

    @staticmethod
    def load_images(img_list):
        return np.asarray(
            [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in img_list],
            dtype=np.uint8,
        )

    def compute_features(self, img_list, batch_size):
        inception_features = []
        for i in tqdm.tqdm(range(0, len(img_list), batch_size)):
            batch_images = self.resize_image(
                self.load_images(img_list[i : i + batch_size])
            )
            inception_features.append(
                self.inception_feature_extractor.get_features(batch_images)
            )
        return np.concatenate(inception_features, axis=0)

    def resize_image(self, images):
        """Resize images using bilinear interpolation."""
        return tf.image.resize(
            images,
            (299, 299),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=True,
            antialias=False,
        )

    def get_single_image(self, path):
        image = cv2.imread(str(path))
        return image.astype(np.uint8)[np.newaxis, :, :, :]

    # this function takes 2 directory and calculates the FID score between them
    def get_fid_from_2_datasets(self, fdir1, fdir2, batch_size):
        dataset_1_features = []
        dataset_2_features = []

        img_list1 = list(sorted(Path(fdir1).glob("*.png")))
        img_list2 = list(sorted(Path(fdir2).glob("*.png")))

        dataset_1_features = self.compute_features(img_list1, batch_size)
        dataset_2_features = self.compute_features(img_list2, batch_size)

        fid = compute_FID(dataset_1_features, dataset_2_features)
        print("FID score:", fid)

        return fid

    def get_batch_images(self, img_list):
        images = []
        for path in img_list:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            images.append(image)
        return np.asarray(images).astype(np.uint8)

    def get_FID_from_random_samples(self, n_images, batch_size=1000):
        generated_inception_features = []
        gt_inception_features = []

        for i in tqdm.tqdm(range(0, n_images, batch_size)):
            random_latents = self.first_stage_model.sample_latent_vector(batch_size)
            generated_images = self.first_stage_model.generate_images(random_latents)
            gt_images = self.load_images(self.test_imgs_list[i : i + batch_size])

            resized_generated = self.resize_image(generated_images)
            resized_gt = self.resize_image(gt_images)

            generated_inception_features.append(
                self.inception_feature_extractor.get_features(resized_generated)
            )
            gt_inception_features.append(
                self.inception_feature_extractor.get_features(resized_gt)
            )

            self.save_images(generated_images, self.test_output / "random_samples")
            self.idx += batch_size  # increment the index

        generated_features = np.concatenate(generated_inception_features, axis=0)
        gt_features = np.concatenate(gt_inception_features, axis=0)

        fid = compute_FID(generated_features, gt_features)
        print("FID score between tests:", fid)
        self.file.write("\get_FID_from_random_samples: " + str(fid))

        return fid

    # example of calculating the frechet inception distance in Keras for cifar10

    def save_images(self, images, path):
        for i, image in enumerate(images):
            cv2.imwrite(str(path) + "/" + str(self.idx + i) + ".png", image)

    def get_fid_wout_finetune(self, n_images, batch_size):
        generated_inception_features = []
        gt_inception_features = []

        for i in range(0, n_images, batch_size):
            print(i, n_images)

            gt_images = self.get_batch_images(self.test_imgs_list[i : i + batch_size])
            latents = self.confignet_model.encode_images(gt_images)
            generated_images = self.confignet_model.generate_images(latents)

            resized_images = {
                "generated": self.resize_image(generated_images),
                "gt": self.resize_image(gt_images),
            }

            generated_inception_features.append(
                self.inception_feature_extractor.get_features(
                    resized_images["generated"]
                )
            )

            gt_inception_features.append(
                self.inception_feature_extractor.get_features(resized_images["gt"])
            )

            self.save_images(
                generated_images, self.test_output / "reconstructed_images"
            )
            self.idx = i

        generated_inception_features = np.concatenate(
            generated_inception_features, axis=0
        )
        gt_inception_features = np.concatenate(gt_inception_features, axis=0)

        fid = compute_FID(generated_inception_features, gt_inception_features)

        print("FID score:", fid)
        self.file.write("\nget_fid_wout_finetune: " + str(fid))
        return fid

    def get_fid(self, n_images):
        # keep track of average FID score
        generated_inception_features = []
        gt_inception_features = []

        for i, a_path in enumerate(self.test_imgs_list[0:n_images]):
            gt_image = self.get_single_image(a_path)
            print(i, n_images)

            latent = self.confignet_model.encode_images(gt_image)

            self.confignet_model.fine_tune_on_img(gt_image, 100)
            generated = self.confignet_model.generator_fine_tuned(latent)
            generated = np.clip(generated, -1.0, 1.0)
            generated = ((generated + 1) * 127.5).astype(np.uint8)

            generated = self.resize_image(generated)
            gt_image = self.resize_image(gt_image)

            generated_inception_features.append(
                self.inception_feature_extractor.get_features(generated)
            )

            gt_inception_features.append(
                self.inception_feature_extractor.get_features(gt_image)
            )

            tf.keras.utils.save_img(
                f"{self.test_output}/finetuned_images/{i}.png",
                generated[0][:, :, ::-1],
            )

            tf.keras.utils.save_img(
                f"{self.test_output}/finetuned_gt/{i}.png",
                gt_image[0][:, :, ::-1],
            )

            self.confignet_model.generator_fine_tuned = None

        generated_inception_features = np.concatenate(
            generated_inception_features, axis=0
        )
        gt_inception_features = np.concatenate(gt_inception_features, axis=0)

        fid = compute_FID(generated_inception_features, gt_inception_features)

        print("FID score:", fid)
        self.file.write("\nget_fid: " + str(fid))
        return fid

    def get_fid_from_latentgan(self, n_images):
        generated_inception_features = []
        gt_inception_features = []

        for i, a_path in enumerate(self.test_imgs_list[0:n_images]):
            gt_image = self.get_single_image(a_path)
            print(i, n_images)
            # random latents from latent gan
            latent = self.latentgan_model.generate_latents(1)

            self.confignet_model.fine_tune_on_img(gt_image, 100)
            generated = self.confignet_model.generator_fine_tuned(latent)
            # min max value

            generated = np.clip(generated, -1.0, 1.0)
            generated = ((generated + 1) * 127.5).astype(np.uint8)

            generated = self.resize_image(generated)
            gt_image = self.resize_image(gt_image)

            generated_inception_features.append(
                self.inception_feature_extractor.get_features(generated)
            )

            gt_inception_features.append(
                self.inception_feature_extractor.get_features(gt_image)
            )

            if Path(self.test_output / "latent_finetuned_images").exists() == False:
                os.makedirs(self.test_output / "latent_finetuned_images")

            tf.keras.utils.save_img(
                f"{self.test_output}/latent_finetuned_images/{i}.png",
                generated[0][:, :, ::-1],
            )

            self.confignet_model.generator_fine_tuned = None

        generated_inception_features = np.concatenate(
            generated_inception_features, axis=0
        )
        gt_inception_features = np.concatenate(gt_inception_features, axis=0)

        fid = compute_FID(generated_inception_features, gt_inception_features)

        print("FID score:", fid)
        self.file.write("\nget_fid_from_latentgan: " + str(fid))
        return fid


def parse_args(args):
    parser = argparse.ArgumentParser(description="ConfigNet training")
    parser.add_argument(
        "--output_dir",
        help="Path to the directory where the output will be stored",
        required=True,
    )
    parser.add_argument(
        "--second_stage_model_path", help="Path to the model to be evaluated"
    )
    parser.add_argument(
        "--first_stage_model_path", help="Path to the model to be evaluated"
    )
    parser.add_argument(
        "--latent_gan_model_path", help="Path to the model to be evaluated"
    )
    parser.add_argument(
        "--test_image_path",
        help="Path to the test images",
        default="/home2/xcnf86/confignet_stylegan/FFHQ_test",
    )

    args = parser.parse_args(args)

    fid_object = FIDTest(
        test_image_path=args.test_image_path,
        confignet_model_path=args.second_stage_model_path,
        first_stage_model_path=args.first_stage_model_path,
        latent_gan_model_path=args.latent_gan_model_path,
    )

    # test1 between datasets
    # CELEBHQ_CLEAN_TRAIN_PATH = "/mnt/6TB/FaceDatasets/CelebHQ/CelebAMask-HQ_256x256_tight_cropped_augmented_clean"
    # score = fid_object.get_fid_from_2_datasets(
    #     fdir1=args.test_image_path,
    #     fdir2=CELEBHQ_CLEAN_TRAIN_PATH,
    #     n_images=9999,
    #     batch_size=50,
    # )

    # test 2 between random samples
    # score = fid_object.get_FID_from_random_samples(
    #     n_images=9999, batch_size=50
    # )  # 37.87 -38.25

    # test 3 between reconstructred samples
    # score = fid_object.get_fid_wout_finetune(n_images=9999, batch_size=50)
    # print("[get_FID_from_random_samples] FID score:", score)

    finetuned = "/mnt/SSD/ConfigNetLight2D/experiments/celeba_gr_percept_g2_d1_latent_gan/finetuned_images"
    # random = "/mnt/SSD/confignet_stylegan/ConfigNetLight2D/experiments/celeba_gr_percept_g2_d1_latent_gan/random_samples"
    # score = fid_object.get_fid_from_2_datasets(
    #     fdir1=args.test_image_path, fdir2=finetuned, batch_size=50
    # )
    score = fid_object.get_fid_from_latentgan(n_images=9999)
    # score = fid_object.get_fid_from_latentgan(n_images=9999, batch_size=50)
    print("[get_FID_from_random_samples] FID score:", score)


if __name__ == "__main__":
    parse_args(sys.argv[1:])
