# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Class for storing training data for neural renderer training"""

import numpy as np
import os
import sys
import cv2
import glob
import pickle
import ntpath
import json
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
import transformations
from typing import List, Dict, Any, Tuple

from . import dataset_utils
from .face_image_normalizer import FaceImageNormalizer
from .metrics import inception_distance


class CustomDataset:
    """Dataset used in the training of CONFIG and other DNNs in this repo"""

    def __init__(self, img_shape: Tuple[int, int, int], is_synthetic: bool):
        self.img_shape = img_shape
        # These things are generated at dataset creation
        self.imgs = None
        self.imgs_memmap_filename = None
        self.imgs_memmap_shape = None
        self.imgs_memmap_dtype = None

        self.inception_features = None
        self.render_metadata = None

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

        image_paths = []
        image_paths.extend(glob.glob(os.path.join(input_dir, "*.png")))

        self._initialize_imgs_memmap(len(image_paths), output_path)

        for i in range(len(image_paths)):
            if i % max([1, (len(image_paths) // 100)]) == 0:
                perc_complete = 100 * i / len(image_paths)
                print("Image reading %d%% complete" % (perc_complete))

            img_filename_with_ext = ntpath.basename(image_paths[i])
            img_filename = img_filename_with_ext.split(".")[0]
            self.imgs[i] = cv2.imread(image_paths[i])

        self._compute_inception_features()
        self.save(output_path)

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

    def _load_metadata(self, image_paths: List[str]) -> Dict[str, Any]:
        image_paths_split = [
            os.path.split(os.path.splitext(path)[0]) for path in image_paths
        ]
        metadata_paths = [
            os.path.join(head_tail[0], "..", "meta" + head_tail[1][3:] + ".json")
            for head_tail in image_paths_split
        ]

        render_metadata = []
        for path in metadata_paths:
            metadata = json.load(open(path))
            render_metadata.append(metadata)

        return render_metadata

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

        return dataset
