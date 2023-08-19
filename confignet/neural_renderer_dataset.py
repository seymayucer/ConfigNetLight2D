# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Class for storing training data for neural renderer training"""

import numpy as np
import pandas as pd
import os
import sys
import cv2
import glob

# import pickle
import pickle5 as pickle
import ntpath
import json
import h5py
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
import transformations
from typing import List, Dict, Any, Tuple
from pathlib import Path
from . import dataset_utils
from .metrics import inception_distance


class OneHotDistribution:
    """A uniform discrete distribution represented as one-hot vector

    Has same interface as scikitlearn's GMM.
    """

    def __init__(self):
        self.n_features = None

    def fit(self, X):
        self.n_features = X.shape[1]

    def sample(self, n_samples=1):
        sampled_indices = np.random.randint(0, self.n_features, size=n_samples)
        one_hot = np.zeros((n_samples, self.n_features), np.float32)
        one_hot[np.arange(n_samples), sampled_indices] = 1

        return one_hot, sampled_indices


class ExemplarDistribution:
    """An arbitrary exemplar-based distribution

    Has same interface as scikitlearn's GMM.
    """

    def __init__(self):
        self.examplars = None
        self.n_exemplars = None

    def fit(self, X):
        self.exemplars = X
        self.n_exemplars = self.exemplars.shape[0]

    def sample(self, n_samples=1):
        sampled_indices = np.random.randint(0, self.n_exemplars, size=n_samples)
        output = self.exemplars[sampled_indices]

        return output, None


class EyeRegionSpec:
    """Specs of the eye region in the UV space of the 3D model used in synthetic data"""

    eye_region_max_y = 0.15
    eye_region_min_y = 0.07

    l_eye_region_max_x = 0.16
    l_eye_region_min_x = 0.09
    r_eye_region_max_x = 0.91
    r_eye_region_min_x = 0.84


class NeuralRendererDataset:
    """Dataset used in the training of CONFIG and other DNNs in this repo"""

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        is_synthetic: bool,
        head_rotation_range=((-30, 30), (-10, 10), (-10, 10)),
        eye_rotation_range=((-25, 25), (-15, 15), (0, 0)),
    ):
        self.img_shape = img_shape
        self.is_synthetic = is_synthetic
        self.head_rotation_range = np.array(head_rotation_range)
        self.eye_rotation_range = np.array(eye_rotation_range)

        # These things are generated at dataset creation
        self.imgs = None
        self.imgs_memmap_filename = None
        self.imgs_memmap_shape = None
        self.imgs_memmap_dtype = None

        self.inception_features = None
        self.render_metadata = None

        # Masks showing location of eye region in the synthetic data
        self.eye_masks = None
        # CelebA attributes
        self.attributes = None

        # These things are generated at training time
        # Processed render_metadata, computed based on DNN config
        self.metadata_inputs = None
        self.metadata_input_distributions = None
        # Labels for each metadat input, for example names of expressions that correspond to various blendshapes
        self.metadata_input_labels = None

    def generate_face_dataset(
        self,
        input_dir: str,
        output_path: str,
        attribute_label_file_path=None,
        pre_normalize=True,
    ) -> None:
        # FaceImageNormalizer.normalize_dataset_dir(
        #     input_dir, pre_normalize, self.img_shape)

        image_paths = []
        image_paths.extend(glob.glob(os.path.join(input_dir, "*.png")))

        if self.is_synthetic:
            metadata = self._load_metadata(image_paths)
            # image_paths, metadata = self._remove_samples_with_out_of_range_pose(
            #     image_paths, metadata
            # )
            self.render_metadata = metadata
            self.eye_masks = []

        if attribute_label_file_path is not None:
            image_attributes = dataset_utils.parse_celeba_attribute_file(
                attribute_label_file_path
            )
            self.attributes = []

        self._initialize_imgs_memmap(len(image_paths), output_path)

        for i in range(len(image_paths)):
            if i % max([1, (len(image_paths) // 100)]) == 0:
                perc_complete = 100 * i / len(image_paths)
                print("Image reading %d%% complete" % (perc_complete))

            img_filename_with_ext = ntpath.basename(image_paths[i])
            img_filename = img_filename_with_ext.split(".")[0]

            if self.attributes is not None:
                self.attributes.append(image_attributes[img_filename])

            self.imgs[i] = cv2.imread(image_paths[i])

        self._compute_inception_features()
        self.save(output_path)

    def generate_synth_face_dataset(
        self, input_dir: str, output_path: str, attribute_label_file_path=None
    ) -> None:
        # df = pd.read_csv(attribute_label_file_path)
        param_folder = Path(input_dir).parent / "parameter_setup_4"
        df = pd.read_csv(param_folder / "attributes_nan_clean.csv")

        total_image_num = df.shape[0]
        print(f"{total_image_num} number of image has found.")

        if self.is_synthetic:
            self.render_metadata = self._load_metadata(param_folder / "parameters.h5")
            self.eye_masks = []

        if attribute_label_file_path is not None and not self.is_synthetic:
            image_attributes = dataset_utils.pd_celeba_attribute_file(
                attribute_label_file_path
            )
            self.attributes = []

        self._initialize_imgs_memmap(df.shape[0], output_path)
        # self._initialize_jsons_mmap(df.shape[0], output_path)

        not_found_no = 0

        for i, row in df.iterrows():
            img_file_name = f'{input_dir}/{row["file"]}.png'
            # json_file_name = f'{input_json_dir}/{row["json_name"]}'

            if Path(img_file_name).exists():
                if i % max([1, (total_image_num // 100)]) == 0:
                    perc_complete = 100 * i / total_image_num
                    print("Image reading %d%% complete" % (perc_complete))

                img_filename_with_ext = ntpath.basename(img_file_name)
                img_filename = img_filename_with_ext.split(".")[0]

                if self.attributes is not None:
                    self.attributes.append(image_attributes[img_filename])

                self.imgs[i] = cv2.imread(img_file_name)

                # if self.is_synthetic:
                #     eye_mask = self._get_eye_mask_for_image_path(
                #         f'{input_dir}/../masks/{row["file"]}_eye_mask.png'
                #     )

                #     self.eye_masks.append(eye_mask)

        print("file reading is done.")
        self._compute_inception_features()
        print("inception scores are obtained.")
        self.save(output_path)
        print("it is generated all is fine")

    def _initialize_imgs_memmap(self, n_images: int, output_path: str) -> None:
        self.imgs_memmap_shape = (n_images, *self.img_shape)

        self.imgs_memmap_dtype = np.uint8
        self.imgs_memmap_filename = (
            os.path.splitext(os.path.basename(output_path))[0] + "_imgs.dat"
        )
        basedir = os.path.dirname(output_path)
        self.imgs = np.memmap(
            os.path.join(basedir, self.imgs_memmap_filename),
            self.imgs_memmap_dtype,
            "w+",
            shape=self.imgs_memmap_shape,
        )

    # def _initialize_eye_masks_memmap(self, n_images: int, output_path: str) -> None:
    #     self.imgs_memmap_shape = (n_images, *self.img_shape[:-1])  # 256x256

    #     self.imgs_memmap_dtype = np.uint8
    #     self.imgs_memmap_filename = (
    #         os.path.splitext(os.path.basename(output_path))[0] + "_eye_masks.dat"
    #     )
    #     basedir = os.path.dirname(output_path)
    #     self.eye_masks = np.memmap(
    #         os.path.join(basedir, self.imgs_memmap_filename),
    #         self.imgs_memmap_dtype,
    #         "w+",
    #         shape=self.imgs_memmap_shape,
    #     )
    #     self.eye_masks = np.array(self.eye_masks)

    def process_metadata(self, config: Dict[str, Any], update_config=False) -> None:
        """Preprocesses self.render_metadata to a format that can be ingested in CONFIG training based on the network config.

        Reads metadata inputs that are to be used from the config.
        If update_config is specified, the method updates the input dimensionality in the config.
        Updates dictionary of metadata input vectors, each vector corresponds to a different metadata type (hair style, texture, etc).
        """

        self.metadata_inputs = {}
        self.metadata_input_distributions = {}
        self.metadata_input_labels = {}

        def fit_distribution(data, distr_type):
            if distr_type == "GMM":
                distr = GaussianMixture()
                distr.fit(data)
            elif distr_type == "one_hot":
                distr = OneHotDistribution()
                distr.fit(data)
            elif distr_type == "exemplar":
                distr = ExemplarDistribution()
                distr.fit(data)

            return distr

        # self.metadata_inputs = self.render_metadata  # done
        # self.metadata_inputs["rotations"] = self.render_metadata["head"]
        # self.metadata_input_labels["rotations"] = None
        # self.metadata_input_distributions["rotations"] = fit_distribution(
        #     self.render_metadata["head"], "exemplar"
        # )
        print(self.render_metadata.keys(), config["facemodel_inputs"].keys())
        for input_name in config["facemodel_inputs"].keys():
            self.render_metadata[input_name] = self.render_metadata[input_name].astype(
                np.float32
            )

            self.metadata_input_labels[input_name] = None
            self.metadata_input_distributions[input_name] = fit_distribution(
                self.render_metadata[input_name], "exemplar"
            )

            # self.render_metadata[input_name][np.isnan(
            #     self.render_metadata[input_name])] = 0.0001

            self.metadata_inputs[input_name] = self.render_metadata[input_name]

            if update_config:
                config["facemodel_inputs"][input_name] = (
                    self.render_metadata[input_name].shape[1:],
                    config["facemodel_inputs"][input_name][1],
                )

    def _load_metadata(self, h5file_path):
        render_metadata = {}
        print(h5file_path)
        hf = h5py.File(h5file_path, "r")
        for keys in hf["meta_names"]:
            keys = keys.decode("UTF8")
            render_metadata[keys] = hf[keys][()]
        hf.close()
        return render_metadata

    def _get_eye_mask_for_image_path(self, image_path: str) -> np.ndarray:
        eye_mask = cv2.imread(image_path, -1)
        eye_mask = eye_mask // 255
        return eye_mask.astype(np.uint8)

    def write_images(self, directory: str, draw_landmarks=False) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(len(self.imgs)):
            if draw_landmarks:
                img = np.copy(self.imgs[i])
                for landmark in self.landmarks[i]:
                    cv2.circle(
                        img, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0)
                    )
            else:
                img = self.imgs[i]

            cv2.imwrite(os.path.join(directory, str(i).zfill(5) + ".jpg"), img)

        mean_img = np.mean(self.imgs, axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(directory, "mean_img.jpg"), mean_img)

    def write_images_by_attribute(self, directory: str) -> None:
        assert self.attributes is not None
        assert all(
            [
                self.attributes[0].keys() == img_attributes.keys()
                for img_attributes in self.attributes
            ]
        )

        attribute_names = self.attributes[0].keys()
        for attribute_name in attribute_names:
            imgs_with_attribute = [
                i
                for i, img_attributes in enumerate(self.attributes)
                if img_attributes[attribute_name]
            ]

            attribute_output_dir = os.path.join(directory, attribute_name)
            if not os.path.exists(attribute_output_dir):
                os.makedirs(attribute_output_dir)
            for img_idx in imgs_with_attribute:
                cv2.imwrite(
                    os.path.join(attribute_output_dir, str(img_idx).zfill(6) + ".jpg"),
                    self.imgs[img_idx],
                )

    def get_attribute_values(
        self, sample_idxs: List[int], attribute_names: List[str]
    ) -> np.ndarray:
        assert self.attributes is not None

        attribute_values = []
        for idx in sample_idxs:
            sample_attributes = self.attributes[idx]
            attribute_present = [
                sample_attributes[attribute_name] for attribute_name in attribute_names
            ]
            attribute_values.append(attribute_present)

        return np.array(attribute_values)

    def _compute_inception_features(self) -> None:
        feature_extractor = inception_distance.InceptionFeatureExtractor(
            self.imgs.shape[1:]
        )
        self.inception_features = feature_extractor.get_features(self.imgs)

    def save(self, filename: str) -> None:
        # Delete the memory-mapped image array so it does not get pickled
        del self.imgs
        self.imgs = None
        pickle.dump(self, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

        # Re-load the image array
        basedir = os.path.dirname(filename)
        self.imgs = np.memmap(
            os.path.join(basedir, self.imgs_memmap_filename),
            self.imgs_memmap_dtype,
            "r",
            shape=self.imgs_memmap_shape,
        )

    @staticmethod
    def load(filename: str) -> "NeuralRendereDataset":
        # Older datasets might not load properly due to changes in repo structure, the code in except fixes it
        try:
            dataset = pickle.load(open(filename, "rb"))
        except:
            from . import neural_renderer_dataset

            sys.modules["neural_renderer_dataset"] = neural_renderer_dataset

            dataset = pickle.load(open(filename, "rb"))

        basedir = os.path.dirname(filename)
        dataset.imgs = np.memmap(
            os.path.join(basedir, dataset.imgs_memmap_filename),
            dataset.imgs_memmap_dtype,
            "r",
            shape=dataset.imgs_memmap_shape,
        )

        # dataset.eye_masks = np.asarray(dataset.eye_masks)
        # if 'val' in filename:
        #     import pdb
        #     pdb.set_trace()

        return dataset
